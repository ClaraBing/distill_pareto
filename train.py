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

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    alpha: float,
    temperature: float,
) -> tuple[torch.Tensor, dict]:
    """Compute training loss.

    For distillation: alpha * CE_loss + (1-alpha) * KL_loss.
    For standard training: CE_loss (alpha is ignored).

    All losses are computed only over answer tokens (labels != -100).
    """
    mask = labels != -100  # [B, L]
    B, L, V = student_logits.shape

    # ── CE loss ───────────────────────────────────────────────────────────────
    # Flatten and apply mask
    flat_logits = student_logits.reshape(-1, V)    # [B*L, V]
    flat_labels = labels.reshape(-1)               # [B*L]
    ce_loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)

    if teacher_logits is None:
        return ce_loss, {"ce_loss": ce_loss.item()}

    # ── KL loss ───────────────────────────────────────────────────────────────
    # Apply temperature scaling; compute only over masked positions
    teacher_logits = teacher_logits.to(student_logits.dtype)
    if temperature != 1.0:
        student_scaled = student_logits / temperature
        teacher_scaled = teacher_logits / temperature
    else:
        student_scaled = student_logits
        teacher_scaled = teacher_logits

    student_log_probs = F.log_softmax(student_scaled, dim=-1)   # [B, L, V]
    teacher_probs = F.softmax(teacher_scaled, dim=-1)           # [B, L, V]

    # Select only answer token positions
    student_lp_masked = student_log_probs[mask]   # [N_ans, V]
    teacher_p_masked = teacher_probs[mask]         # [N_ans, V]

    # KL(teacher || student) = sum_x teacher(x) * (log teacher(x) - log student(x))
    kl_loss = F.kl_div(student_lp_masked, teacher_p_masked, reduction="batchmean")

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
        f"{cfg.training.mode}_{cfg.model.family}_{cfg.model.size}_{cfg.data.task}"
    )
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        name=run_name,
        tags=list(cfg.wandb.tags),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # ── model & tokenizer ─────────────────────────────────────────────────────
    model = get_model(
        family=cfg.model.family,
        size=cfg.model.size,
        variant=cfg.model.variant,
        torch_dtype=cfg.model.torch_dtype,
        device_map=cfg.model.device_map,
    )
    tokenizer = get_tokenizer(cfg.model.family, cfg.model.size, cfg.model.variant)

    # If device_map="auto" the model is already placed; otherwise move manually.
    if cfg.model.device_map != "auto":
        model = model.to(device)

    # ── data ──────────────────────────────────────────────────────────────────
    train_loader = get_loader(
        file_paths=list(cfg.data.train_files),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    eval_loader = get_loader(
        file_paths=list(cfg.data.eval_files),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

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
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
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

    # Track the batch offset into the full dataset for aligning teacher logits
    sample_offset = 0

    for epoch in range(cfg.training.n_epochs):
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            B = input_ids.size(0)

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
                labels=labels,
                teacher_logits=teacher_batch,
                alpha=alpha,
                temperature=temperature,
            )

            # Gradient accumulation
            loss = loss / cfg.training.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % cfg.training.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.training.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                log_dict["lr"] = scheduler.get_last_lr()[0]
                log_dict["epoch"] = epoch + batch_idx / len(train_loader)
                wandb.log(log_dict, step=global_step)

                # ── periodic evaluation ───────────────────────────────────────
                if global_step % cfg.training.eval_interval == 0:
                    model.eval()
                    metrics = run_eval(
                        model=model,
                        loader=eval_loader,
                        device=device,
                    )
                    wandb.log(
                        {f"eval/{k}": v for k, v in metrics.items() if isinstance(v, float)},
                        step=global_step,
                    )
                    print(
                        f"[step {global_step}] eval loss={metrics['loss']:.4f} "
                        f"acc={metrics['token_accuracy']:.4f}"
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
    final_metrics = run_eval(
        model=model,
        loader=eval_loader,
        device=device,
    )
    wandb.log(
        {f"final/{k}": v for k, v in final_metrics.items() if isinstance(v, float)}
    )
    print("Final evaluation:")
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
