"""
Model architectures for document field extraction.

Core arms:
    - TransformerExtractor  (Arm A: 6 attention layers)
    - MambaExtractor        (Arm B: 6 Mamba SSM layers)
    - HybridExtractor       (Arm C: configurable Mamba:Attention ratio)

Hybrid ablation variants: 5:1, 4:2, 3:3, 2:4
"""

from .base import BaseExtractor
from .transformer import TransformerExtractor
from .mamba_model import MambaExtractor
from .hybrid import HybridExtractor


MODEL_REGISTRY = {
    "transformer": TransformerExtractor,
    "mamba": MambaExtractor,
    "hybrid_5_1": HybridExtractor,
    "hybrid_4_2": HybridExtractor,
    "hybrid_3_3": HybridExtractor,
    "hybrid_2_4": HybridExtractor,
}


def build_model(arm: str, **kwargs) -> BaseExtractor:
    """
    Factory function — build a model by arm name.

    Args:
        arm: one of the keys in MODEL_REGISTRY
        **kwargs: passed to the model constructor

    Returns:
        Instantiated model (on CPU — caller moves to device)
    """
    if arm not in MODEL_REGISTRY:
        raise ValueError(f"Unknown arm '{arm}'. Choose from: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[arm](**kwargs)
