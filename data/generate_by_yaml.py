"""Generate every synthetic task from a single YAML spec.

Reads a YAML file describing the hyperparameters for all tasks plus the model
families to tokenize for, then writes everything under
``data_files/{yaml_file_name_without_extension}/``:

  - ``{task}_{split}.jsonl``        one per task, for split in {train, eval}
  - ``{model_family}_train.pt``     tokenized train set, merged over all tasks
  - ``{model_family}_{task}_eval.pt`` tokenized eval set, one file per task
  - a copy of the input YAML file

Tasks with 0 samples for a split are skipped (no JSONL, no .pt).

The tokenizer is shared across all sizes within a model family, so a single
representative size per family (see ``_FAMILY_TOKENIZER``) is used to produce
the ``.pt`` files.

Usage:
    python -m data.generate_by_yaml configs/data_gen/all_tasks.yaml
"""

import shutil
import sys
import tempfile
from pathlib import Path
from typing import List

from omegaconf import DictConfig, OmegaConf

from data import SyntheticTask, merge_tokenized_files, tokenize
from models.get_model import get_tokenizer

SPLITS = ("train", "eval")

# Representative (size, variant) per family. The tokenizer is identical across
# sizes within a family, so the resulting token ids do not depend on this choice.
_FAMILY_TOKENIZER = {
    "qwen3": ("0.6B", "base"),
    "smollm2": ("135M", "base"),
}


def build_task(task_name: str, params: DictConfig, split: str) -> SyntheticTask:
    """Instantiate the right SyntheticTask for *task_name* and *split*."""
    seed = params.seed_train if split == "train" else params.seed_eval

    if task_name == "memorization":
        from data.memorization import MemorizationTask
        return MemorizationTask(
            n_pairs=params.n_pairs,
            min_T=params.min_T,
            max_T=params.max_T,
            pool_seed=params.pool_seed,
            seed=seed,
        )

    elif task_name == "line_graph":
        from data.line_graph import LineGraphTask
        return LineGraphTask(
            min_T=params.min_T,
            max_T=params.max_T,
            seed=seed,
        )

    elif task_name == "s5":
        from data.s5 import S5Task
        noun_seed = (
            params.noun_seed_train if split == "train" else params.noun_seed_eval
        )
        action_probs = (
            list(params.action_probs) if params.get("action_probs") else None
        )
        return S5Task(
            n_actions=params.n_actions,
            action_probs=action_probs,
            min_T=params.min_T,
            max_T=params.max_T,
            noun_seed=noun_seed,
            seed=seed,
        )

    else:
        raise ValueError(f"Unknown task '{task_name}'")


def main(yaml_path: str) -> None:
    cfg = OmegaConf.load(yaml_path)

    out_dir = Path("data_files") / Path(yaml_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep a copy of the spec alongside the data it produced.
    shutil.copy(yaml_path, out_dir / Path(yaml_path).name)

    task_names: List[str] = list(cfg.tasks.keys())
    max_length = cfg.max_length

    # 1) Generate per-task JSONL for every split.
    # Track which (task, split) actually produced samples, so tasks set to
    # 0 samples (e.g. a "memory only" spec) are skipped end to end rather than
    # tokenized into empty/merge-incompatible tensors.
    generated: set = set()
    for task_name in task_names:
        params = cfg.tasks[task_name]
        for split in SPLITS:
            n_samples = (
                params.n_samples_train if split == "train" else params.n_samples_eval
            )
            if n_samples <= 0:
                print(f"Skipping '{task_name}' {split} (n_samples=0)")
                continue
            task = build_task(task_name, params, split)
            print(f"Generating {n_samples} {split} samples for '{task_name}'…")
            samples = task.generate_samples(n_samples)
            jsonl_path = out_dir / f"{task_name}_{split}.jsonl"
            task.save_samples(samples, str(jsonl_path))
            print(f"  Saved JSONL → {jsonl_path}")
            generated.add((task_name, split))

    # 2) Tokenize with each family's tokenizer. Train is merged over all tasks
    #    into {family}_train.pt; eval is kept per-task as {family}_{task}_eval.pt
    #    so each task can be evaluated separately.
    for family in cfg.model_families:
        if family not in _FAMILY_TOKENIZER:
            raise ValueError(
                f"Unknown model family '{family}'. "
                f"Supported: {list(_FAMILY_TOKENIZER)}."
            )
        size, variant = _FAMILY_TOKENIZER[family]
        tokenizer_name = get_tokenizer(family, size, variant).name_or_path

        # Train: one file merged over all tasks.
        train_tasks = [t for t in task_names if (t, "train") in generated]
        train_path = out_dir / f"{family}_train.pt"
        if not train_tasks:
            print(f"Skipping {family}_train.pt (no train samples)")
        elif len(train_tasks) == 1:
            # Single task → tokenize straight to the final path. Going through
            # a temp file + merge would force a torch.cat copy, doubling peak
            # memory for large datasets.
            tokenize(
                jsonl_path=str(out_dir / f"{train_tasks[0]}_train.jsonl"),
                output_path=str(train_path),
                tokenizer_name_or_path=tokenizer_name,
                max_length=max_length,
            )
            print(f"  Saved → {train_path}")
        else:
            with tempfile.TemporaryDirectory() as tmp:
                per_task_pts = []
                for task_name in train_tasks:
                    pt_path = Path(tmp) / f"{task_name}.pt"
                    tokenize(
                        jsonl_path=str(out_dir / f"{task_name}_train.jsonl"),
                        output_path=str(pt_path),
                        tokenizer_name_or_path=tokenizer_name,
                        max_length=max_length,
                    )
                    per_task_pts.append(str(pt_path))
                merge_tokenized_files(per_task_pts, output_path=str(train_path))
                print(f"  Merged → {train_path}")

        # Eval: one tokenized file per task (not merged).
        eval_tasks = [t for t in task_names if (t, "eval") in generated]
        if not eval_tasks:
            print(f"Skipping {family}_*_eval.pt (no eval samples)")
        for task_name in eval_tasks:
            eval_path = out_dir / f"{family}_{task_name}_eval.pt"
            tokenize(
                jsonl_path=str(out_dir / f"{task_name}_eval.jsonl"),
                output_path=str(eval_path),
                tokenizer_name_or_path=tokenizer_name,
                max_length=max_length,
            )
            print(f"  Saved → {eval_path}")

    print(f"\nDone. Output written to {out_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python -m data.generate_by_yaml <config.yaml>",
            file=sys.stderr,
        )
        sys.exit(1)
    main(sys.argv[1])
