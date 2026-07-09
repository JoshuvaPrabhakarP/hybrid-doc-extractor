"""
On-demand model loading and GPU-decorated inference for the Gradio + ZeroGPU app.

ZeroGPU dynamically attaches a real NVIDIA H200 GPU only for the duration of
a function decorated with @spaces.GPU, then releases it. This means all 6
architectures (including the CUDA-only Mamba/Hybrid arms) work exactly as
they do locally — no degradation, unlike a plain CPU deployment.
"""

import sys
from pathlib import Path
from functools import lru_cache

import torch
import yaml
from tokenizers import Tokenizer

try:
    import spaces  # Hugging Face ZeroGPU SDK
    ZEROGPU_AVAILABLE = True
except ImportError:
    # Allows local testing without the `spaces` package installed.
    # @spaces.GPU becomes a no-op decorator in that case.
    ZEROGPU_AVAILABLE = False
    class _NoOpSpaces:
        @staticmethod
        def GPU(func):
            return func
    spaces = _NoOpSpaces()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model
from training.dataset import LABEL2ID, ID2LABEL, NUM_LABELS
from training.utils import load_checkpoint

ARM_DISPLAY_NAMES = {
    "transformer": "Pure Transformer",
    "mamba": "Pure Mamba",
    "hybrid_5_1": "Hybrid 5:1 (Mamba:Attention)",
    "hybrid_4_2": "Hybrid 4:2 (Mamba:Attention)",
    "hybrid_3_3": "Hybrid 3:3 (Mamba:Attention)",
    "hybrid_2_4": "Hybrid 2:4 (Mamba:Attention)",
}

# Hardcoded class weights from Phase 5 training (avoids shipping train.jsonl
# just to recompute these — see training/dataset.py compute_class_weights()).
# Order matches LABEL2ID: O, B-DATE, I-DATE, B-AMOUNT, I-AMOUNT, B-NAME, I-NAME
FALLBACK_CLASS_WEIGHTS = [0.16, 10.02, 16.32, 3.89, 84.50, 9.87, 7.77]


@lru_cache(maxsize=1)
def load_config() -> dict:
    cfg_path = PROJECT_ROOT / "configs" / "experiment.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_tokenizer() -> Tokenizer:
    cfg = load_config()
    tok_path = PROJECT_ROOT / cfg["tokenizer"]["path"]
    return Tokenizer.from_file(str(tok_path))


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


@lru_cache(maxsize=6)
def load_model(arm_name: str, seed: int = 42) -> torch.nn.Module:
    """
    Load a single architecture's best checkpoint on demand.

    lru_cache means each architecture is only loaded from disk once per
    process lifetime — switching back to a previously-used arm is instant.

    Note: the model is moved to 'cuda' here, but under ZeroGPU this doesn't
    actually grab a GPU yet — torch.cuda calls are intercepted/deferred by
    ZeroGPU's patching until code actually runs inside an @spaces.GPU call.
    """
    cfg = load_config()
    class_weights = torch.tensor(FALLBACK_CLASS_WEIGHTS)

    ckpt_path = PROJECT_ROOT / "checkpoints" / f"{arm_name}_seed{seed}" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model_kwargs = get_model_kwargs(cfg, arm_name, class_weights)
    model = build_model(arm_name, **model_kwargs)
    load_checkpoint(ckpt_path, model)
    model.eval()

    if torch.cuda.is_available():
        model = model.to("cuda")

    return model


@spaces.GPU
def _run_inference(model, input_ids, attention_mask, fsw_mask):
    """
    The actual GPU-bound forward pass, isolated in its own function so
    ZeroGPU only attaches a GPU for this specific call, not the whole
    request (tokenization, formatting, etc. run on CPU as normal).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    fsw_mask = fsw_mask.to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask, first_subword_mask=fsw_mask)

    return {
        "word_preds": output["word_preds"].cpu(),
        "word_mask": output["word_mask"].cpu(),
    }


def extract_entities_from_text(text: str, arm_name: str, seed: int = 42) -> list[dict]:
    """
    Run live inference on raw text and return extracted entities.

    Returns:
        List of dicts: [{"word": str, "tag": str, "word_idx": int}, ...]
    """
    cfg = load_config()
    tokenizer = load_tokenizer()
    model = load_model(arm_name, seed)

    words = text.split()
    if not words:
        return []

    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")

    sub_ids = [cls_id]
    fsw_mask = [False]

    for word in words:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        piece_ids = encoded.ids or [tokenizer.token_to_id("[UNK]")]
        for i, pid in enumerate(piece_ids):
            sub_ids.append(pid)
            fsw_mask.append(i == 0)

    sub_ids.append(sep_id)
    fsw_mask.append(False)

    max_len = cfg["model"]["max_len"]
    sub_ids = sub_ids[:max_len]
    fsw_mask = fsw_mask[:max_len]

    input_ids = torch.tensor([sub_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    fsw_tensor = torch.tensor([fsw_mask], dtype=torch.bool)

    result = _run_inference(model, input_ids, attention_mask, fsw_tensor)

    word_preds = result["word_preds"][0]
    word_mask = result["word_mask"][0]
    n_words_predicted = word_mask.sum().item()

    results = []
    for i in range(min(n_words_predicted, len(words))):
        tag_id = word_preds[i].item()
        tag = ID2LABEL.get(tag_id, "O")
        results.append({"word": words[i], "tag": tag, "word_idx": i})

    return results


def entities_to_highlighted_format(results: list[dict]) -> list[tuple[str, str | None]]:
    """
    Convert extracted entities into gr.HighlightedText's expected format:
    a list of (text_chunk, label_or_None) tuples.
    """
    formatted = []
    for r in results:
        word = r["word"]
        tag = r["tag"]
        if tag == "O":
            formatted.append((word + " ", None))
        else:
            field = tag.split("-")[-1]  # DATE, AMOUNT, or NAME
            formatted.append((word + " ", field))
    return formatted


def entities_to_field_dict(results: list[dict]) -> dict[str, str]:
    """Group consecutive same-field tags into the final extracted field strings."""
    extracted = {"DATE": [], "AMOUNT": [], "NAME": []}
    current_field = None
    current_words = []

    for r in results:
        tag = r["tag"]
        if tag.startswith("B-"):
            if current_field:
                extracted[current_field].append(" ".join(current_words))
            current_field = tag[2:]
            current_words = [r["word"]]
        elif tag.startswith("I-") and current_field == tag[2:]:
            current_words.append(r["word"])
        else:
            if current_field:
                extracted[current_field].append(" ".join(current_words))
            current_field = None
            current_words = []
    if current_field:
        extracted[current_field].append(" ".join(current_words))

    return {k: ", ".join(v) if v else "—" for k, v in extracted.items()}
