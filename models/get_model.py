"""Model and tokenizer loading for Qwen3 and SmolLM2 families."""

from typing import Optional
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

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
    from_pretrained: bool = True,
    embedding_from_pretrained: bool = False,
    **kwargs,
) -> PreTrainedModel:
    """Load and return a causal LM from the registry.

    Args:
        family: 'qwen3' or 'smollm2'.
        size: model size string (e.g. '0.6B', '360M').
        variant: 'base' or 'instruct'.
        torch_dtype: dtype string or torch.dtype; defaults to bfloat16.
        device_map: passed to from_pretrained; 'auto' spreads across GPUs.
        from_pretrained: if True, load the pretrained weights; if False, build
            the model from the pretrained *config* only (random initialization).
        embedding_from_pretrained: only used when from_pretrained=False. If True,
            copy the pretrained input embedding and output (unembedding) matrices
            into the otherwise randomly-initialized model.
        **kwargs: forwarded to AutoModelForCausalLM.from_pretrained.
    """
    model_id = _resolve_model_id(family, size, variant)

    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)

    if from_pretrained:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **kwargs,
        )
    else:
        # Architecture from the pretrained config, but randomly initialized
        # weights. from_config does not accept device_map; the caller places
        # the model on-device.
        config = AutoConfig.from_pretrained(model_id, **kwargs)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)
        if embedding_from_pretrained:
            _copy_pretrained_embeddings(model, model_id, torch_dtype, **kwargs)
    return model


def _copy_pretrained_embeddings(
    model: PreTrainedModel,
    model_id: str,
    torch_dtype: torch.dtype,
    **kwargs,
) -> None:
    """Copy input/output embeddings from the pretrained checkpoint into `model`.

    Loads the pretrained model on CPU, copies its input embedding and output
    (unembedding) weights/bias into the (random-init) `model`, then frees it.
    When the architecture ties embeddings, copying the input matrix already sets
    the output projection; the explicit output copy below is a harmless no-op.
    """
    src = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, device_map=None, **kwargs
    )

    model.get_input_embeddings().weight.data.copy_(
        src.get_input_embeddings().weight.data
    )

    dst_out, src_out = model.get_output_embeddings(), src.get_output_embeddings()
    if dst_out is not None and src_out is not None:
        dst_out.weight.data.copy_(src_out.weight.data)
        if getattr(dst_out, "bias", None) is not None and getattr(src_out, "bias", None) is not None:
            dst_out.bias.data.copy_(src_out.bias.data)

    del src


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
