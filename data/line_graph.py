"""Line-graph reasoning task: infer taller/shorter from a chain of comparisons."""

import random
from typing import List, Dict

from data import SyntheticTask, NAMES


class LineGraphTask(SyntheticTask):
    """Reasoning over an ordered chain of height comparisons.

    The ground-truth ordering is a line graph 0 < 1 < ... < T-1 (left = taller).
    Edges are presented in shuffled order with randomly oriented phrasing.
    """

    # Fixed class-level name pool
    ALL_NAMES: List[str] = NAMES

    def __init__(self, min_T: int = 3, max_T: int = 10, seed: int = 42):
        if max_T > len(self.ALL_NAMES):
            raise ValueError(
                f"max_T={max_T} exceeds the name pool size {len(self.ALL_NAMES)}"
            )
        self.min_T = min_T
        self.max_T = max_T
        self.rng = random.Random(seed)

    def generate_samples(self, n_samples: int) -> List[Dict[str, str]]:
        samples = []
        for _ in range(n_samples):
            samples.append(self._make_sample())
        return samples

    def _make_sample(self) -> Dict[str, str]:
        T = self.rng.randint(self.min_T, self.max_T)

        # Draw T unique names; their list index gives the height order (0 = tallest)
        names = self.rng.sample(self.ALL_NAMES, T)

        # Build the T-1 edges of the line graph, shuffle their presentation order
        edges = list(range(T - 1))  # edge i connects label i and i+1
        self.rng.shuffle(edges)

        statements = []
        for i in edges:
            a_name = names[i]       # label i  → taller end
            b_name = names[i + 1]   # label i+1 → shorter end
            # Randomly phrase the edge as "A is taller than B" or "B is shorter than A"
            if self.rng.random() < 0.5:
                statements.append(f"{a_name} is taller than {b_name}")
            else:
                statements.append(f"{b_name} is shorter than {a_name}")

        # Sample two distinct names and ask for their relative height
        x_idx, y_idx = self.rng.sample(range(T), 2)
        x_name = names[x_idx]
        y_name = names[y_idx]

        prefix = ". ".join(statements) + "."
        inp = f"{prefix} Is {x_name} taller than {y_name}?"
        out = "Yes" if x_idx < y_idx else "No"

        return {"input": inp, "output": out}
