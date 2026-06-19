"""Memorization task: in-weight memorization of (A, B) ownership relations.

Train and validation/test sets share the same pool of `n_pairs` (A, B) pairs
(controlled by `pool_seed`), so the model must memorize each pair into its
weights — rather than recover it from in-context evidence — to answer the
held-out query at the end of each sample.
"""

import random
import string
from typing import List, Dict, Tuple

from data import SyntheticTask, ADJECTIVES, NOUNS

_TOKEN_ALPHABET = string.ascii_lowercase + string.digits


def _make_phrase(rng: random.Random) -> str:
    """Sample an '{adj}-{noun}-{rnd}' phrase with an 8-char random suffix."""
    adj = rng.choice(ADJECTIVES)
    noun = rng.choice(NOUNS)
    suffix = ''.join(rng.choices(_TOKEN_ALPHABET, k=8))
    return f"{adj}-{noun}-{suffix}"


class MemorizationTask(SyntheticTask):
    """In-weight memorization of (A owns B) relations."""

    def __init__(
        self,
        n_pairs: int = 100,
        min_T: int = 2,
        max_T: int = 10,
        pool_seed: int = 0,
        seed: int = 42,
    ):
        if min_T < 2:
            raise ValueError(f"min_T must be >= 2 (got {min_T})")
        if max_T < min_T:
            raise ValueError(f"max_T={max_T} must be >= min_T={min_T}")
        if n_pairs < max_T:
            raise ValueError(
                f"n_pairs={n_pairs} must be >= max_T={max_T} so each sample "
                "can draw T distinct pairs"
            )

        self.n_pairs = n_pairs
        self.min_T = min_T
        self.max_T = max_T
        self.rng = random.Random(seed)

        # The pool is built from `pool_seed` alone, so instantiating the task
        # with the same `pool_seed` (and any `seed`) yields the same phrases.
        pool_rng = random.Random(pool_seed)
        self.pairs: List[Tuple[str, str]] = [
            (_make_phrase(pool_rng), _make_phrase(pool_rng))
            for _ in range(n_pairs)
        ]

    def generate_samples(self, n_samples: int) -> List[Dict[str, str]]:
        # Guarantee every pair is written out as a context relation at least
        # once across the dataset — otherwise the model could never learn it
        # (a pair seen only as a query never has its relation stated). Each
        # sample contributes T-1 >= 1 context slots, so n_samples >= n_pairs
        # is sufficient for full coverage.
        if n_samples < self.n_pairs:
            raise ValueError(
                f"n_samples={n_samples} must be >= n_pairs={self.n_pairs} so "
                "every pair can appear as a context relation at least once"
            )

        # Indices not yet used as a context relation; handed out to context
        # slots before any random fill. Shared (and consumed) across samples.
        uncovered = list(range(self.n_pairs))
        self.rng.shuffle(uncovered)
        return [self._make_sample(uncovered) for _ in range(n_samples)]

    def _make_sample(self, uncovered: List[int]) -> Dict[str, str]:
        """Build one sample, consuming still-uncovered pairs from *uncovered*."""
        T = self.rng.randint(self.min_T, self.max_T)
        n_context = T - 1  # min_T >= 2 guarantees n_context >= 1

        # Fill the T-1 context slots with still-uncovered pairs first, then top
        # up with random distinct pairs.
        context_indices: List[int] = []
        while uncovered and len(context_indices) < n_context:
            context_indices.append(uncovered.pop())
        if len(context_indices) < n_context:
            chosen = set(context_indices)
            pool = [i for i in range(self.n_pairs) if i not in chosen]
            context_indices.extend(
                self.rng.sample(pool, n_context - len(context_indices))
            )
        # Shuffle so forced-covered pairs aren't biased toward the first slot.
        self.rng.shuffle(context_indices)

        # Query pair: distinct from the context pairs, so it requires
        # memorization rather than in-context lookup.
        query_pool = [i for i in range(self.n_pairs) if i not in set(context_indices)]
        query_index = self.rng.choice(query_pool)

        context_pairs = [self.pairs[i] for i in context_indices]
        query_a, query_b = self.pairs[query_index]

        statements = []
        for a, b in context_pairs:
            if self.rng.random() < 0.5:
                statements.append(f"{a} owns {b}")
            else:
                statements.append(f"{b} is owned by {a}")

        if self.rng.random() < 0.5:
            question = f"What does {query_a} own?"
            answer = query_b
        else:
            question = f"Who owns {query_b}?"
            answer = query_a

        prefix = ". ".join(statements)
        inp = f"{prefix}. {question}" if prefix else question
        return {"input": inp, "output": answer}
