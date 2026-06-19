"""Evaluation script for distillation experiments.

Can be imported in train.py or run directly:
    python evaluate.py data=s5 model=qwen3_0.6b
"""

import os
from pathlib import Path
from typing import Dict, Optional

import hydra
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from data import get_loader
from models.get_model import get_model, get_tokenizer


# ── core eval function (importable) ──────────────────────────────────────────

@torch.no_grad()
def eval(
    model: PreTrainedModel,
    loader: DataLoader,
    device: str | torch.device = "cuda",
    save_generations: bool = False,
    eval_generations: bool = False,
    save_logits: bool = False,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    max_new_tokens: int = 64,
    output_path: Optional[str] = None,
    answer_only_loss: bool = False,
) -> Dict[str, float | torch.Tensor]:
    """Evaluate *model* on *loader* and return a metrics dict.

    Args:
        model: the model to evaluate.
        loader: DataLoader yielding {input_ids, attention_mask, labels}.
        device: device to run on.
        save_generations: if True, perform greedy decoding and save text outputs.
            Requires *tokenizer* and *output_path*.
        eval_generations: if True, 'sequence_accuracy' is computed from
            free-running greedy generation — the model conditions on its own
            previous outputs (the prompt only is given) rather than on the
            ground-truth answer prefix (teacher forcing). Requires *tokenizer*.
            This is the faithful end-to-end accuracy; the teacher-forced number
            is misleadingly high (see note below).
        save_logits: if True, save per-token logits for distillation.
            Requires *output_path*. Logits are stored in float16 to save space.
        tokenizer: needed when save_generations or eval_generations is True.
        max_new_tokens: max tokens generated for save/eval generations.
        output_path: directory path for saved files.
        answer_only_loss: if True, compute loss over answer tokens only
            (positions where labels != -100); otherwise all non-pad tokens.

    Returns:
        dict with at minimum 'loss' and 'sequence_accuracy' keys. When
        eval_generations=True, 'sequence_accuracy' is generation-based;
        otherwise it is teacher-forced.
        When save_logits=True, also includes 'logits_path'.
        When save_generations=True, also includes 'generations_path'.
    """
    if (save_generations or eval_generations) and tokenizer is None:
        raise ValueError(
            "save_generations and eval_generations require a tokenizer"
        )

    model.eval()
    model.to(device)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    # Generation-based exact-match counters (used when eval_generations=True).
    total_gen_correct = 0
    total_gen_samples = 0

    all_logits: list[torch.Tensor] = []
    all_generations: list[str] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # [B, L, V]

        # Autoregressive shift: position i's logits predict token i+1.
        shift_logits = logits[:, :-1, :]
        V = shift_logits.size(-1)

        # Loss: answer-only tokens or all non-pad tokens, per config.
        if answer_only_loss:
            loss_targets = labels[:, 1:]  # -100 for prompt/pad, token id for answer
        else:
            loss_targets = input_ids[:, 1:].masked_fill(
                ~attention_mask[:, 1:].bool(), -100
            )
        loss = F.cross_entropy(
            shift_logits.reshape(-1, V),
            loss_targets.reshape(-1),
            ignore_index=-100,
        )
        total_loss += loss.item() * input_ids.size(0)

        # Sequence accuracy: ALL answer tokens must be correct (exact match).
        # Teacher-forced token accuracy is misleadingly high at init because the
        # model sees the correct answer prefix and can predict structural tokens
        # (e.g. "Slot", "contains", ",") without knowing the task content.
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        preds = shift_logits.argmax(dim=-1)
        has_answer = mask.any(dim=-1)
        all_correct = ((preds == shift_labels) | ~mask).all(dim=-1)
        total_correct += (all_correct & has_answer).sum().item()
        total_tokens += has_answer.sum().item()

        if save_logits:
            all_logits.append(logits.cpu().to(torch.float16))

        if save_generations or eval_generations:
            # Free-running greedy decode of the answer portion for each sample.
            # The model sees ONLY the prompt (everything up to the first answer
            # token, i.e. the first label != -100) and generates autoregressively
            # from its own outputs — no teacher forcing. A mistake on an early
            # slot therefore propagates into the context for later slots.
            for i in range(input_ids.size(0)):
                answer_idx = (labels[i] != -100).nonzero(as_tuple=True)[0]
                if len(answer_idx) == 0:
                    continue
                start = answer_idx[0]
                prompt_ids = input_ids[i, :start].unsqueeze(0)
                prompt_mask = attention_mask[i, :start].unsqueeze(0)
                gen_ids = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
                # Decode only the newly generated tokens.
                new_ids = gen_ids[0, prompt_ids.size(1):]
                gen_text = tokenizer.decode(new_ids, skip_special_tokens=True)

                if save_generations:
                    all_generations.append(gen_text)

                if eval_generations:
                    # Exact-match the generated answer against the reference
                    # answer (the answer tokens from labels). Compare decoded
                    # text so trailing eos/pad differences don't matter.
                    ref_text = tokenizer.decode(
                        labels[i, answer_idx], skip_special_tokens=True
                    )
                    total_gen_correct += int(gen_text.strip() == ref_text.strip())
                    total_gen_samples += 1

    n_samples = len(loader.dataset)
    if eval_generations:
        sequence_accuracy = total_gen_correct / max(total_gen_samples, 1)
    else:
        sequence_accuracy = total_correct / max(total_tokens, 1)
    metrics: Dict = {
        "loss": total_loss / n_samples,
        "sequence_accuracy": sequence_accuracy,
    }

    if output_path is not None:
        Path(output_path).mkdir(parents=True, exist_ok=True)

    if save_logits and all_logits:
        logits_tensor = torch.cat(all_logits, dim=0)  # [N, L, V]
        logits_file = os.path.join(output_path, "teacher_logits.pt")
        torch.save(logits_tensor, logits_file)
        metrics["logits_path"] = logits_file
        print(f"Saved teacher logits {tuple(logits_tensor.shape)} → {logits_file}")

    if save_generations and all_generations:
        gen_file = os.path.join(output_path, "generations.txt")
        with open(gen_file, "w") as f:
            for gen in all_generations:
                f.write(gen + "\n")
        metrics["generations_path"] = gen_file
        print(f"Saved {len(all_generations)} generations → {gen_file}")

    return metrics


# ── CLI entry point ───────────────────────────────────────────────────────────

@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── wandb ─────────────────────────────────────────────────────────────────
    run_name = cfg.wandb.name or (
        f"eval_{cfg.model.family}_{cfg.model.size}"
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

    # ── data ──────────────────────────────────────────────────────────────────
    eval_loader = get_loader(
        file_paths=list(cfg.data.eval_files),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    save_logits = cfg.get("save_logits", False)
    save_generations = cfg.get("save_generations", False)
    eval_generations = cfg.get("eval_generations", False)
    output_path = cfg.get("eval_output_path", "eval_outputs")
    answer_only_loss = cfg.training.answer_only_loss

    metrics = eval(
        model=model,
        loader=eval_loader,
        device=device,
        save_generations=save_generations,
        eval_generations=eval_generations,
        save_logits=save_logits,
        tokenizer=tokenizer,
        output_path=output_path,
        answer_only_loss=answer_only_loss,
    )

    print("Evaluation results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    wandb.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    wandb.finish()


if __name__ == "__main__":
    main()
