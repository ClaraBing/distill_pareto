"""S5 group task: track permutation state of 5 objects through a sequence of actions."""

import random
from itertools import combinations
from typing import List, Dict, Tuple

from data import SyntheticTask, NOUNS


# ── action helpers ─────────────────────────────────────────────────────────────

Action = Tuple  # ("swap", i, j)  or  ("shift",)


def _apply_action(state: List[int], action: Action) -> List[int]:
    """Return a new state after applying *action* to *state*."""
    state = state[:]
    if action[0] == "swap":
        _, i, j = action
        state[i], state[j] = state[j], state[i]
    else:  # shift left
        state = state[1:] + [state[0]]
    return state


def _action_to_text(action: Action) -> str:
    """Convert an action to natural-language (1-indexed slots)."""
    if action[0] == "swap":
        _, i, j = action
        return f"swaps objects in slot {i + 1} and slot {j + 1}"
    return "shifts all the objects to the left by 1 slot"


# ── task class ────────────────────────────────────────────────────────────────

class S5Task(SyntheticTask):
    """Symmetric-group S5 permutation tracking task.

    Train and test sets may use different 100-noun subsets and different action
    probability distributions to probe out-of-distribution generalisation.
    """

    ALL_NOUNS: List[str] = NOUNS

    def __init__(
        self,
        n_actions: int = 5,
        action_probs: List[float] | None = None,
        min_T: int = 3,
        max_T: int = 10,
        noun_seed: int = 42,
        seed: int = 0,
    ):
        """
        Args:
            n_actions: total number of generator actions (≥ 2).
            action_probs: probability for each action; uniform if None.
            min_T / max_T: range of sequence lengths (inclusive).
            noun_seed: controls which 100 nouns are sampled from the 500-pool.
            seed: RNG seed for sample generation.
        """
        if n_actions < 2:
            raise ValueError("n_actions must be at least 2")

        self.min_T = min_T
        self.max_T = max_T
        self.rng = random.Random(seed)
        noun_rng = random.Random(noun_seed)

        # Sample the working 100-noun set
        self.nouns: List[str] = noun_rng.sample(list(self.ALL_NOUNS), 100)

        # Build generator list
        self.actions: List[Action] = [
            ("swap", 0, 1),     # fixed: swap first two
            ("shift",),         # fixed: left shift
        ]
        # Forbidden swap: (0,1) is already the first action
        forbidden = {(0, 1), (1, 0)}
        all_swaps = list(combinations(range(5), 2))
        available_swaps = [s for s in all_swaps if s not in forbidden]
        extra = noun_rng.sample(available_swaps, n_actions - 2)
        for i, j in extra:
            self.actions.append(("swap", i, j))

        if action_probs is None:
            n = len(self.actions)
            self.action_probs = [1.0 / n] * n
        else:
            if len(action_probs) != len(self.actions):
                raise ValueError(
                    f"action_probs length {len(action_probs)} != n_actions {n_actions}"
                )
            total = sum(action_probs)
            self.action_probs = [p / total for p in action_probs]

    def generate_samples(self, n_samples: int) -> List[Dict[str, str]]:
        return [self._make_sample() for _ in range(n_samples)]

    def _make_sample(self) -> Dict[str, str]:
        T = self.rng.randint(self.min_T, self.max_T)

        # Sample T actions
        action_seq: List[Action] = self.rng.choices(
            self.actions, weights=self.action_probs, k=T
        )

        # Sample 5 distinct objects for the initial slots
        objects: List[str] = self.rng.sample(self.nouns, 5)

        # Simulate: state[i] = index into *objects* currently in slot i
        # TODO: can speed up to O(log T) if necessary.
        state = [0, 1, 2, 3, 4]
        for action in action_seq:
            state = _apply_action(state, action)

        # Build natural-language prompt
        obj_str = ", ".join(objects)
        intro = (
            f"There are 5 slots, containing {obj_str} "
            f"from slot 1 to 5, respectively."
        )

        action_sentences = [_action_to_text(a) for a in action_seq]
        if len(action_sentences) == 1:
            action_text = f"Someone {action_sentences[0]}."
        else:
            middle = ". ".join(
                f"Someone then {s}" for s in action_sentences[1:-1]
            )
            first = f"Someone {action_sentences[0]}."
            last = f"Finally, someone {action_sentences[-1]}."
            parts = [first]
            if middle:
                parts.append(middle + ".")
            parts.append(last)
            action_text = " ".join(parts)

        question = "What objects do the 5 slots contain?"
        inp = f"{intro} {action_text} {question}"

        # Build answer from final state
        final_objects = [objects[i] for i in state]
        answer_parts = [
            f"Slot {slot + 1} contains {obj}"
            for slot, obj in enumerate(final_objects)
        ]
        # Join with commas, Oxford comma before "and" on the last item
        if len(answer_parts) == 1:
            out = answer_parts[0] + "."
        else:
            out = ", ".join(answer_parts[:-1]) + f", and {answer_parts[-1]}."

        return {"input": inp, "output": out}
