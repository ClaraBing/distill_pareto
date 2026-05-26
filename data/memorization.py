"""Memorization task: learn (A owns B) associations from query-answer pairs."""

import random
import string
from typing import List, Dict

from data import SyntheticTask, ADJECTIVES, NOUNS

_TOKEN_ALPHABET = string.ascii_lowercase + string.digits


def _make_phrase(rng: random.Random) -> str:
    """Sample an 'adjective noun token' phrase with a random 8-char token."""
    adj = rng.choice(ADJECTIVES)
    noun = rng.choice(NOUNS)
    random_token = ''.join(rng.choices(_TOKEN_ALPHABET, k=8))
    return f"{adj} {noun} {random_token}"


class MemorizationTask(SyntheticTask):
    """Memorization of (A owns B) associations queried in two directions."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_samples(self, n_samples: int) -> List[Dict[str, str]]:
        """Generate *n_samples* (A, B) pairs with random query direction."""
        samples = []

        for _ in range(n_samples):
            a = _make_phrase(self.rng)
            b = _make_phrase(self.rng)

            # Randomly choose query direction
            if self.rng.random() < 0.5:
                inp = f"What does {a} own?"
                out = f"{a} owns {b}."
            else:
                inp = f"Who owns {b}?"
                out = f"{b} is owned by {a}."

            samples.append({"input": inp, "output": out})

        return samples
