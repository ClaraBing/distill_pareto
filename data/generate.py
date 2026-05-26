"""CLI for generating and tokenizing synthetic datasets.

Usage:
    python -m data.generate task=memorization data=memorization model=qwen3_0.6b
    python -m data.generate task=s5          data=s5          model=qwen3_0.6b
    python -m data.generate task=line_graph  data=line_graph  model=qwen3_0.6b
"""

import hydra
from omegaconf import DictConfig
from pathlib import Path

from data import tokenize
from models.get_model import get_tokenizer


def _build_task(cfg: DictConfig, split: str):
    """Instantiate the right SyntheticTask for *split* ('train' or 'eval')."""
    task_name = cfg.data.task

    if task_name == "memorization":
        from data.memorization import MemorizationTask
        seed = cfg.data.seed_train if split == "train" else cfg.data.seed_eval
        return MemorizationTask(seed=seed)

    elif task_name == "line_graph":
        from data.line_graph import LineGraphTask
        seed = cfg.data.seed_train if split == "train" else cfg.data.seed_eval
        return LineGraphTask(
            min_T=cfg.data.min_T,
            max_T=cfg.data.max_T,
            seed=seed,
        )

    elif task_name == "s5":
        from data.s5 import S5Task
        noun_seed = cfg.data.noun_seed_train if split == "train" else cfg.data.noun_seed_eval
        seed = cfg.data.seed_train if split == "train" else cfg.data.seed_eval
        action_probs = list(cfg.data.action_probs) if cfg.data.action_probs else None
        return S5Task(
            n_actions=cfg.data.n_actions,
            action_probs=action_probs,
            min_T=cfg.data.min_T,
            max_T=cfg.data.max_T,
            noun_seed=noun_seed,
            seed=seed,
        )

    else:
        raise ValueError(f"Unknown task '{task_name}'")


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    tokenizer = get_tokenizer(cfg.model.family, cfg.model.size, cfg.model.variant)
    tokenizer_name = tokenizer.name_or_path
    data_dir = Path(cfg.data.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "eval"):
        n_samples = (
            cfg.data.n_samples_train if split == "train" else cfg.data.n_samples_eval
        )
        task = _build_task(cfg, split)
        print(f"Generating {n_samples} {split} samples for '{cfg.data.task}'…")
        samples = task.generate_samples(n_samples)

        jsonl_path = str(data_dir / f"{split}.jsonl")
        pt_path = str(data_dir / f"{split}.pt")

        task.save_samples(samples, jsonl_path)
        print(f"  Saved JSONL → {jsonl_path}")

        tokenize(
            jsonl_path=jsonl_path,
            output_path=pt_path,
            tokenizer_name_or_path=tokenizer_name,
            max_length=cfg.data.max_length,
        )


if __name__ == "__main__":
    main()
