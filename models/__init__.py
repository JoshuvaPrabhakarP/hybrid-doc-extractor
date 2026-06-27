"""
Model architectures for document field extraction.

Three arms for controlled comparison:
    - TransformerExtractor  (Arm A: 6 attention layers)
    - MambaExtractor        (Arm B: 6 Mamba SSM layers)
    - HybridExtractor       (Arm C: 5 Mamba + 1 attention)
"""

from .base import BaseExtractor
from .transformer import TransformerExtractor
from .mamba_model import MambaExtractor
from .hybrid import HybridExtractor


MODEL_REGISTRY = {
    "transformer": TransformerExtractor,
    "mamba": MambaExtractor,
    "hybrid_5_1": HybridExtractor,
}


def build_model(arm: str, **kwargs) -> BaseExtractor:
    """
    Factory function — build a model by arm name.

    Args:
        arm: one of "transformer", "mamba", "hybrid_5_1"
        **kwargs: passed to the model constructor

    Returns:
        Instantiated model (on CPU — caller moves to device)
    """
    if arm not in MODEL_REGISTRY:
        raise ValueError(f"Unknown arm '{arm}'. Choose from: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[arm](**kwargs)
