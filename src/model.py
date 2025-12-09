"""
Model loading and tokenizer preparation
"""

import torch
from transformer_lens import HookedTransformer


def load_model(
    model_name: str = "EleutherAI/pythia-160m-deduped",
    fallback: str = "EleutherAI/pythia-70m-deduped",
    device: str = "cuda",
) -> HookedTransformer:
    """
    Load model from HF with fallback support.
    Returns model (tokenizer accessible via model.tokenizer).
    """
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            dtype=dtype,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )
        return model
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        try:
            print(f"Trying fallback model: {fallback}")
            model = HookedTransformer.from_pretrained(
                fallback,
                device=device,
                dtype=dtype,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
            )
            return model
        except Exception as e2:
            print(f"Failed to load fallback model {fallback}: {e2}")
            raise
