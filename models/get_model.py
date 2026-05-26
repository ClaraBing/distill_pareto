"""Model and tokenizer loading for Qwen3 and SmolLM2 families."""

from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

# ── model registry ─────────────────────────────────────────────────────────────

_QWEN3: dict[str, str] = {
    "0.6B": "Qwen/Qwen3-0.6B",
    "1.7B": "Qwen/Qwen3-1.7B",
    "4B":   "Qwen/Qwen3-4B",
    "8B":   "Qwen/Qwen3-8B",
}

_SMOLLM2: dict[str, str] = {
    "135M": "HuggingFaceTB/SmolLM2-135M",
    "360M": "HuggingFaceTB/SmolLM2-360M",
    "1.7B": "HuggingFaceTB/SmolLM2-1.7B",
}


def _resolve_model_id(family: str, size: str, variant: str) -> str:
    """Return the HuggingFace Hub model ID for the requested configuration."""
    family_lower = family.lower()
    if family_lower == "qwen3":
        registry = _QWEN3
        instruct_suffix = "-Instruct"
    elif family_lower == "smollm2":
        registry = _SMOLLM2
        instruct_suffix = "-Instruct"
    else:
        raise ValueError(
            f"Unknown model family '{family}'. Supported: qwen3, smollm2."
        )

    if size not in registry:
        raise ValueError(
            f"Unknown size '{size}' for family '{family}'. "
            f"Supported: {list(registry.keys())}."
        )

    model_id = registry[size]
    if variant == "instruct":
        model_id = model_id + instruct_suffix
    elif variant != "base":
        raise ValueError(f"Unknown variant '{variant}'. Supported: base, instruct.")

    return model_id


def get_model(
    family: str,
    size: str,
    variant: str = "base",
    torch_dtype: str | torch.dtype = "bfloat16",
    device_map: Optional[str] = "auto",
    **kwargs,
) -> PreTrainedModel:
    """Load and return a causal LM from the registry.

    Args:
        family: 'qwen3' or 'smollm2'.
        size: model size string (e.g. '0.6B', '360M').
        variant: 'base' or 'instruct'.
        torch_dtype: dtype string or torch.dtype; defaults to bfloat16.
        device_map: passed to from_pretrained; 'auto' spreads across GPUs.
        **kwargs: forwarded to AutoModelForCausalLM.from_pretrained.
    """
    model_id = _resolve_model_id(family, size, variant)

    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        **kwargs,
    )
    return model


def get_tokenizer(
    family: str,
    size: str,
    variant: str = "base",
) -> PreTrainedTokenizerBase:
    """Load and return the tokenizer for the requested model.

    Sets pad_token to eos_token when it is missing (common for causal LMs).
    """
    model_id = _resolve_model_id(family, size, variant)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
