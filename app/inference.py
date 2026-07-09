"""
On-demand model loading and inference for the Streamlit dashboard.

Models are loaded lazily and cached via st.cache_resource so switching
between architectures in the UI doesn't reload from disk every time,
but nothing loads until the user actually selects it.
"""

import sys
from pathlib import Path

import streamlit as st
import torch
import yaml
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model
from training.dataset import LABEL2ID, ID2LABEL, NUM_LABELS, compute_class_weights
from training.utils import load_checkpoint

ARM_DISPLAY_NAMES = {
    "transformer": "Pure Transformer",
    "mamba": "Pure Mamba",
    "hybrid_5_1": "Hybrid 5:1 (Mamba:Attention)",
    "hybrid_4_2": "Hybrid 4:2 (Mamba:Attention)",
    "hybrid_3_3": "Hybrid 3:3 (Mamba:Attention)",
    "hybrid_2_4": "Hybrid 2:4 (Mamba:Attention)",
}


@st.cache_data
def load_config() -> dict:
    cfg_path = PROJECT_ROOT / "configs" / "experiment.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_tokenizer() -> Tokenizer:
    cfg = load_config()
    tok_path = PROJECT_ROOT / cfg["tokenizer"]["path"]
    return Tokenizer.from_file(str(tok_path))


@st.cache_resource
def load_class_weights(_device: str) -> torch.Tensor:
    """Underscore prefix on _device tells st.cache_resource not to hash it
    as part of the cache key (device string is small/stable, but the
    convention avoids Streamlit trying to hash unrelated objects)."""
    train_jsonl = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
    return compute_class_weights(train_jsonl).to(_device)


def get_model_kwargs(cfg: dict, arm_name: str, class_weights: torch.Tensor) -> dict:
    """Build model constructor kwargs from config (mirrors training/train.py)."""
    model_cfg = cfg["model"]
    kwargs = {
        "vocab_size": cfg["tokenizer"]["vocab_size"],
        "d_model": model_cfg["d_model"],
        "num_labels": NUM_LABELS,
        "max_len": model_cfg["max_len"],
        "dropout": model_cfg["dropout"],
        "pad_token_id": 0,
        "n_layers": model_cfg["n_layers"],
        "use_crf": True,
        "class_weights": class_weights,
    }
    if arm_name == "transformer":
        kwargs["n_heads"] = model_cfg["n_heads"]
        kwargs["d_ff"] = model_cfg["d_ff"]
    elif arm_name == "mamba":
        kwargs["d_state"] = model_cfg["mamba"]["d_state"]
        kwargs["d_conv"] = model_cfg["mamba"]["d_conv"]
        kwargs["expand"] = model_cfg["mamba"]["expand"]
    elif arm_name.startswith("hybrid_"):
        del kwargs["n_layers"]
        kwargs["mamba_layers"] = model_cfg["arms"][arm_name]["mamba_layers"]
        kwargs["attn_layers"] = model_cfg["arms"][arm_name]["attn_layers"]
        kwargs["n_heads"] = model_cfg["n_heads"]
        kwargs["d_ff"] = model_cfg["d_ff"]
        kwargs["d_state"] = model_cfg["mamba"]["d_state"]
        kwargs["d_conv"] = model_cfg["mamba"]["d_conv"]
        kwargs["expand"] = model_cfg["mamba"]["expand"]
    return kwargs


@st.cache_resource(show_spinner=False)
def load_model(arm_name: str, seed: int = 42) -> torch.nn.Module:
    """
    Load a single architecture's best checkpoint on demand.

    Cached by (arm_name, seed) — first selection in the UI triggers the
    actual disk load; subsequent selections of the same arm are instant.
    """
    cfg = load_config()
    device = cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu"
    class_weights = load_class_weights(device)

    ckpt_path = PROJECT_ROOT / "checkpoints" / f"{arm_name}_seed{seed}" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model_kwargs = get_model_kwargs(cfg, arm_name, class_weights)
    model = build_model(arm_name, **model_kwargs).to(device)
    load_checkpoint(ckpt_path, model)
    model.eval()
    return model


def extract_entities_from_text(
    text: str,
    arm_name: str,
    seed: int = 42,
) -> list[dict]:
    """
    Run live inference on raw text and return extracted entities.

    Args:
        text: raw receipt/invoice text (will be whitespace-tokenized)
        arm_name: which architecture to use
        seed: which seed's checkpoint to load

    Returns:
        List of dicts: [{"word": str, "tag": str, "start_idx": int}, ...]
        One entry per whitespace-split word, with its predicted BIO tag.
    """
    cfg = load_config()
    device = cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer()
    model = load_model(arm_name, seed)

    words = text.split()
    if not words:
        return []

    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")

    sub_ids = [cls_id]
    fsw_mask = [False]
    word_boundaries = []  # which word each subword belongs to (-1 for special tokens)

    for word_idx, word in enumerate(words):
        encoded = tokenizer.encode(word, add_special_tokens=False)
        piece_ids = encoded.ids or [tokenizer.token_to_id("[UNK]")]
        for i, pid in enumerate(piece_ids):
            sub_ids.append(pid)
            fsw_mask.append(i == 0)

    sub_ids.append(sep_id)
    fsw_mask.append(False)

    # Truncate to model's max length
    max_len = cfg["model"]["max_len"]
    sub_ids = sub_ids[:max_len]
    fsw_mask = fsw_mask[:max_len]

    input_ids = torch.tensor([sub_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    fsw_tensor = torch.tensor([fsw_mask], dtype=torch.bool, device=device)

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask, first_subword_mask=fsw_tensor)

    word_preds = output["word_preds"][0]
    word_mask = output["word_mask"][0]
    n_words_predicted = word_mask.sum().item()

    results = []
    for i in range(min(n_words_predicted, len(words))):
        tag_id = word_preds[i].item()
        tag = ID2LABEL.get(tag_id, "O")
        results.append({"word": words[i], "tag": tag, "word_idx": i})

    return results
