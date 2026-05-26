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
    save_logits: bool = False,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    max_new_tokens: int = 64,
    output_path: Optional[str] = None,
) -> Dict[str, float | torch.Tensor]:
    """Evaluate *model* on *loader* and return a metrics dict.

    Args:
        model: the model to evaluate.
        loader: DataLoader yielding {input_ids, attention_mask, labels}.
        device: device to run on.
        save_generations: if True, perform greedy decoding and save text outputs.
            Requires *tokenizer* and *output_path*.
        save_logits: if True, save per-token logits for distillation.
            Requires *output_path*. Logits are stored in float16 to save space.
        tokenizer: needed only when save_generations=True.
        max_new_tokens: max tokens generated when save_generations=True.
        output_path: directory path for saved files.

    Returns:
        dict with at minimum 'loss' and 'token_accuracy' keys.
        When save_logits=True, also includes 'logits_path'.
        When save_generations=True, also includes 'generations_path'.
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    all_logits: list[torch.Tensor] = []
    all_generations: list[str] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        logits = outputs.logits  # [B, L, V]

        total_loss += loss.item() * input_ids.size(0)

        # Token accuracy over answer positions only
        mask = labels != -100
        preds = logits.argmax(dim=-1)
        total_correct += (preds[mask] == labels[mask]).sum().item()
        total_tokens += mask.sum().item()

        if save_logits:
            all_logits.append(logits.cpu().to(torch.float16))

        if save_generations and tokenizer is not None:
            # Greedy decode the answer portion for each sample in the batch.
            # Find where the answer starts: the last non-padded input token
            # plus 1 (i.e. the first label != -100).
            for i in range(input_ids.size(0)):
                answer_start = (labels[i] != -100).nonzero(as_tuple=True)[0]
                if len(answer_start) == 0:
                    continue
                prompt_ids = input_ids[i, : answer_start[0]].unsqueeze(0)
                prompt_mask = attention_mask[i, : answer_start[0]].unsqueeze(0)
                gen_ids = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
                # Decode only the newly generated tokens
                new_ids = gen_ids[0, prompt_ids.size(1):]
                all_generations.append(tokenizer.decode(new_ids, skip_special_tokens=True))

    n_samples = len(loader.dataset)
    metrics: Dict = {
        "loss": total_loss / n_samples,
        "token_accuracy": total_correct / max(total_tokens, 1),
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
        f"eval_{cfg.model.family}_{cfg.model.size}_{cfg.data.task}"
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
    output_path = cfg.get("eval_output_path", "eval_outputs")

    metrics = eval(
        model=model,
        loader=eval_loader,
        device=device,
        save_generations=save_generations,
        save_logits=save_logits,
        tokenizer=tokenizer,
        output_path=output_path,
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
