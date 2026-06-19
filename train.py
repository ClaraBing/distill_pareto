"""Training script for language model distillation experiments.

Supports two modes (set via cfg.training.mode):
  - 'standard':     cross-entropy loss against ground-truth labels.
  - 'distillation': alpha * CE + (1-alpha) * KL against pre-saved teacher logits.

Run:
    python train.py                              # default config
    python train.py model=qwen3_4b data=s5      # override model/data
    python train.py training.mode=distillation \\
        distillation.teacher_logits_path=<path>
"""

import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from transformers import get_linear_schedule_with_warmup

from data import get_loader
from evaluate import eval as run_eval
from models.get_model import get_model, get_tokenizer


# ── helpers ────────────────────────────────────────────────────────────────────

def get_memory_stats() -> dict:
    """Return current CUDA memory counters in GiB for wandb logging.

    Covers current and step-peak usage for allocated, reserved, and active
    memory pools, plus the number of cudaMalloc retries (non-zero means the
    caching allocator had to free blocks to satisfy an allocation — a leading
    indicator of OOM pressure). Call torch.cuda.reset_peak_memory_stats()
    after logging so peak values reflect only the most recent step.
    """
    stats = torch.cuda.memory_stats()
    GiB = 1024 ** 3
    return {
        "mem/allocated_GiB":      stats.get("allocated_bytes.all.current", 0) / GiB,
        "mem/peak_allocated_GiB": stats.get("allocated_bytes.all.peak",    0) / GiB,
        "mem/reserved_GiB":       stats.get("reserved_bytes.all.current",  0) / GiB,
        "mem/peak_reserved_GiB":  stats.get("reserved_bytes.all.peak",     0) / GiB,
        "mem/active_GiB":         stats.get("active_bytes.all.current",    0) / GiB,
        "mem/peak_active_GiB":    stats.get("active_bytes.all.peak",       0) / GiB,
        "mem/alloc_retries":      stats.get("num_alloc_retries",           0),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_loss(
    student_logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    alpha: float,
    temperature: float,
    labels: torch.Tensor | None = None,
    answer_only_loss: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Compute training loss (autoregressive).

    For distillation: alpha * CE_loss + (1-alpha) * KL_loss.
    For standard training: CE_loss (alpha is ignored).

    When answer_only_loss=True, both CE and KL terms are restricted to
    positions where labels != -100 (answer tokens only). Otherwise, all
    non-pad positions contribute. logits[:, i] predicts token i+1.
    """
    B, L, V = student_logits.shape

    # Autoregressive shift: position i's logits predict token i+1.
    shift_logits = student_logits[:, :-1, :]        # [B, L-1, V]
    shift_targets = input_ids[:, 1:]                # [B, L-1]

    if answer_only_loss and labels is not None:
        # labels has -100 for prompt/pad tokens, token id for answer tokens
        valid = (labels[:, 1:] != -100)
        ce_targets = labels[:, 1:]                  # -100 already masks non-answer
    else:
        valid = attention_mask[:, 1:].bool()        # all non-pad tokens
        ce_targets = shift_targets.masked_fill(~valid, -100)

    # ── CE loss ─────────────────────────────────────────────────────────────
    ce_loss = F.cross_entropy(
        shift_logits.reshape(-1, V),
        ce_targets.reshape(-1),
        ignore_index=-100,
    )

    if teacher_logits is None:
        return ce_loss, {"ce_loss": ce_loss.item()}

    # ── KL loss ─────────────────────────────────────────────────────────────
    # Match student/teacher distributions at every valid position. Index down
    # to valid positions BEFORE softmax to keep the [N, V] tensors small.
    shift_teacher = teacher_logits[:, :-1, :]       # [B, L-1, V]
    student_sel = shift_logits[valid]               # [N, V]
    teacher_sel = shift_teacher[valid].to(student_sel.dtype)
    if temperature != 1.0:
        student_sel = student_sel / temperature
        teacher_sel = teacher_sel / temperature

    student_lp = F.log_softmax(student_sel, dim=-1)
    teacher_p = F.softmax(teacher_sel, dim=-1)

    # KL(teacher || student) = sum_x teacher(x) * (log teacher(x) - log student(x))
    kl_loss = F.kl_div(student_lp, teacher_p, reduction="batchmean")

    # Temperature-squared rescaling (standard in KD literature)
    if temperature != 1.0:
        kl_loss = kl_loss * (temperature ** 2)

    total_loss = alpha * ce_loss + (1.0 - alpha) * kl_loss

    return total_loss, {
        "loss": total_loss.item(),
        "ce_loss": ce_loss.item(),
        "kl_loss": kl_loss.item(),
    }


# ── main ───────────────────────────────────────────────────────────────────────

@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── wandb ─────────────────────────────────────────────────────────────────
    run_name = cfg.wandb.name or (
        f"{cfg.training.mode}_{cfg.model.family}_{cfg.model.size}"
    )
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        name=run_name,
        tags=list(cfg.wandb.tags),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # ── model & tokenizer ─────────────────────────────────────────────────────
    from_pretrained = bool(cfg.training.from_pretrained)
    embedding_from_pretrained = bool(cfg.training.embedding_from_pretrained)
    model = get_model(
        family=cfg.model.family,
        size=cfg.model.size,
        variant=cfg.model.variant,
        torch_dtype=cfg.model.torch_dtype,
        device_map=cfg.model.device_map,
        from_pretrained=from_pretrained,
        embedding_from_pretrained=embedding_from_pretrained,
    )
    tokenizer = get_tokenizer(cfg.model.family, cfg.model.size, cfg.model.variant)
    if not from_pretrained:
        msg = "Initialized {}/{} from config only (random weights".format(
            cfg.model.family, cfg.model.size
        )
        msg += ", pretrained embeddings)." if embedding_from_pretrained else ")."
        print(msg)

    # With device_map="auto" the pretrained model is already placed; otherwise
    # (manual device_map, or config-only init which ignores device_map) move it.
    if cfg.model.device_map != "auto" or not from_pretrained:
        model = model.to(device)

    # ── freeze layers ─────────────────────────────────────────────────────────
    if cfg.training.freeze_embeddings:
        model.get_input_embeddings().requires_grad_(False)
        print("Froze input embeddings.")
    if cfg.training.freeze_lm_head:
        out_emb = model.get_output_embeddings()
        if out_emb is not None:
            out_emb.requires_grad_(False)
        print("Froze lm_head.")

    # ── data ──────────────────────────────────────────────────────────────────
    train_loader = get_loader(
        file_paths=list(cfg.data.train_files),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )

    # One loader per eval file so metrics are reported separately per file.
    eval_files = list(cfg.data.eval_files)
    if len(eval_files) == 1:
        eval_loaders = {
            "eval": get_loader(
                file_paths=eval_files,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
            )
        }
    else:
        # Eval files are named "{model_family}_{task}_eval.pt"; key each loader
        # by just the task, stripping the model-family prefix and "_eval" suffix.
        model_prefix = f"{cfg.model.family}_"

        def _loader_name(f: str) -> str:
            name = Path(f).stem
            if name.startswith(model_prefix):
                name = name[len(model_prefix):]
            if name.endswith("_eval"):
                name = name[: -len("_eval")]
            return name

        eval_loaders = {
            _loader_name(f): get_loader(
                file_paths=[f],
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
            )
            for f in eval_files
        }

    # ── teacher logits (distillation only) ───────────────────────────────────
    teacher_logits_all: torch.Tensor | None = None
    if cfg.training.mode == "distillation":
        if cfg.distillation.teacher_logits_path is None:
            raise ValueError(
                "distillation.teacher_logits_path must be set in distillation mode"
            )
        teacher_logits_all = torch.load(
            cfg.distillation.teacher_logits_path, weights_only=True
        )
        print(
            f"Loaded teacher logits: {tuple(teacher_logits_all.shape)} "
            f"from {cfg.distillation.teacher_logits_path}"
        )

    # ── optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=(cfg.training.beta1, cfg.training.beta2),
    )

    steps_per_epoch = len(train_loader) // cfg.training.gradient_accumulation_steps
    total_steps = steps_per_epoch * cfg.training.n_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=total_steps,
    )

    # ── training loop ─────────────────────────────────────────────────────────
    global_step = 0
    model.train()

    alpha = cfg.distillation.alpha
    temperature = cfg.distillation.temperature
    answer_only_loss = cfg.training.answer_only_loss
    eval_generations = cfg.get("eval_generations", False)
    save_generations = cfg.get("save_generations", False)

    # Track the batch offset into the full dataset for aligning teacher logits
    sample_offset = 0

    if device == "cuda":
        torch.cuda.memory._record_memory_history(max_entries=100_000)

    for epoch in range(cfg.training.n_epochs):
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            B = input_ids.size(0)

            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                student_logits = outputs.logits  # [B, L, V]

                # Slice teacher logits aligned with this batch
                teacher_batch: torch.Tensor | None = None
                if teacher_logits_all is not None:
                    teacher_batch = teacher_logits_all[
                        sample_offset: sample_offset + B
                    ].to(device)

                loss, log_dict = compute_loss(
                    student_logits=student_logits,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    teacher_logits=teacher_batch,
                    alpha=alpha,
                    temperature=temperature,
                    labels=labels,
                    answer_only_loss=answer_only_loss,
                )

                # Gradient accumulation
                loss = loss / cfg.training.gradient_accumulation_steps
                loss.backward()

            except torch.cuda.OutOfMemoryError:
                snapshot_path = str(output_dir / "oom_snapshot.pkl")
                torch.cuda.memory._dump_snapshot(snapshot_path)
                torch.cuda.memory._record_memory_history(enabled=None)
                wandb.save(snapshot_path)
                raise RuntimeError(
                    f"CUDA OOM at epoch={epoch} batch={batch_idx} "
                    f"(shape={tuple(input_ids.shape)}). "
                    f"Memory snapshot uploaded to wandb and saved to {snapshot_path}. "
                    f"Download and visualize at https://pytorch.org/memory_viz"
                ) from None

            if (batch_idx + 1) % cfg.training.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.training.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                log_dict["lr"] = scheduler.get_last_lr()[0]
                log_dict["epoch"] = epoch + batch_idx / len(train_loader)
                if cfg.training.log_memory and device == "cuda":
                    log_dict.update(get_memory_stats())
                    torch.cuda.reset_peak_memory_stats()
                wandb.log(log_dict, step=global_step)

                # ── periodic evaluation ───────────────────────────────────────
                if global_step % cfg.training.eval_interval == 0:
                    model.eval()
                    for loader_name, loader in eval_loaders.items():
                        metrics = run_eval(
                            model=model,
                            loader=loader,
                            device=device,
                            answer_only_loss=answer_only_loss,
                            eval_generations=eval_generations,
                            save_generations=save_generations,
                            tokenizer=tokenizer,
                            output_path=(
                                str(output_dir / "generations" / f"{loader_name}_step{global_step}")
                                if save_generations else None
                            ),
                        )
                        prefix = "eval" if len(eval_loaders) == 1 else f"eval/{loader_name}"
                        wandb.log(
                            {f"{prefix}/{k}": v for k, v in metrics.items() if isinstance(v, float)},
                            step=global_step,
                        )
                        print(
                            f"[step {global_step}] {loader_name} "
                            f"loss={metrics['loss']:.4f} acc={metrics['sequence_accuracy']:.4f}"
                        )
                    model.train()

                # ── periodic checkpoint ───────────────────────────────────────
                if global_step % cfg.training.save_interval == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"Saved checkpoint → {ckpt_dir}")

            sample_offset += B

        # Reset offset each epoch (teacher logits are aligned per-epoch)
        sample_offset = 0

    # ── final evaluation & save ───────────────────────────────────────────────
    model.eval()
    for loader_name, loader in eval_loaders.items():
        final_metrics = run_eval(
            model=model,
            loader=loader,
            device=device,
            answer_only_loss=answer_only_loss,
            eval_generations=eval_generations,
            save_generations=save_generations,
            tokenizer=tokenizer,
            output_path=(
                str(output_dir / "generations" / f"{loader_name}_final")
                if save_generations else None
            ),
        )
        prefix = "final" if len(eval_loaders) == 1 else f"final/{loader_name}"
        wandb.log(
            {f"{prefix}/{k}": v for k, v in final_metrics.items() if isinstance(v, float)}
        )
        print(f"Final evaluation ({loader_name}):")
        for k, v in final_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final model → {final_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()
