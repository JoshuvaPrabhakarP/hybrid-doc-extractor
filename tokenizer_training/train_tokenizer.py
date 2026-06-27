#!/usr/bin/env python3
"""
Phase 3 — Train a domain-specific BPE tokenizer on receipt/invoice text.

Usage:
    cd ~/hybrid-doc-extractor
    python tokenizer_training/train_tokenizer.py

Reads:   data/processed/train.jsonl
Writes:  tokenizer_training/tokenizer.json   (full HF-compatible tokenizer)
         tokenizer_training/vocab.json        (token → id mapping)
         tokenizer_training/merges.txt        (BPE merge rules)

Design decisions:
  - vocab_size=8000: small enough for a from-scratch model, large enough to cover
    receipt-domain vocabulary (dates, currencies, company names, item descriptions)
  - No lowercasing: case carries NER signal ("TOTAL" vs item text)
  - Whitespace pre-tokenizer: our data is already word-tokenized in train.jsonl
  - Special tokens: [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4
"""

import json
import sys
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_JSONL  = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
OUTPUT_DIR   = PROJECT_ROOT / "tokenizer_training"

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
VOCAB_SIZE     = 8_000


def extract_corpus(train_path: Path) -> list[str]:
    """Read train.jsonl → list of text lines (one per document)."""
    lines = []
    with open(train_path) as f:
        for raw in f:
            rec = json.loads(raw)
            # Join pre-tokenized words back into a single string
            lines.append(" ".join(rec["tokens"]))
    return lines


def train_bpe(corpus_lines: list[str], output_dir: Path) -> Tokenizer:
    """Train a BPE tokenizer on the corpus and save artifacts."""
    # Initialize empty BPE model with [UNK] fallback
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # Pre-tokenizer: split on whitespace (our data is already word-level)
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Trainer config
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,           # subwords must appear ≥2 times
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # cover all bytes
    )

    # Train from in-memory iterator (no temp file needed for 2,947 docs)
    tokenizer.train_from_iterator(corpus_lines, trainer=trainer)

    # Post-processor: add [CLS] ... [SEP] template for compatibility
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )

    # Save all artifacts
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full tokenizer (single-file, HF-compatible)
    tok_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tok_path))

    # 2. Separate vocab.json and merges.txt (classic BPE pair)
    model_files = tokenizer.model.save(str(output_dir))
    # model.save() writes vocab.json and merges.txt to the directory
    print(f"  Saved: {tok_path}")
    for f in model_files:
        print(f"  Saved: {f}")

    return tokenizer


def verify_tokenizer(tokenizer: Tokenizer, corpus_lines: list[str]):
    """Print verification stats and sample tokenizations."""
    vocab = tokenizer.get_vocab()
    print(f"\n{'='*60}")
    print(f"  Tokenizer Verification")
    print(f"{'='*60}")
    print(f"  Vocab size:      {len(vocab)}")
    print(f"  Special tokens:  {SPECIAL_TOKENS}")

    # --- Coverage: tokenize all training docs, count UNKs ---
    total_subwords = 0
    total_unks = 0
    subword_counts = []
    unk_id = tokenizer.token_to_id("[UNK]")

    for line in corpus_lines:
        enc = tokenizer.encode(line)
        # Exclude [CLS] and [SEP] added by post-processor
        ids = enc.ids[1:-1]
        total_subwords += len(ids)
        total_unks += ids.count(unk_id)
        subword_counts.append(len(ids))

    avg_subwords = sum(subword_counts) / len(subword_counts)
    max_subwords = max(subword_counts)
    unk_rate = (total_unks / total_subwords * 100) if total_subwords > 0 else 0

    print(f"\n  Coverage (train set):")
    print(f"    Total subword tokens: {total_subwords:,}")
    print(f"    [UNK] tokens:         {total_unks:,} ({unk_rate:.2f}%)")
    print(f"    Avg subwords/doc:     {avg_subwords:.1f}")
    print(f"    Max subwords/doc:     {max_subwords}")

    # --- Sample tokenizations ---
    samples = [
        "TOTAL $ 12.50",
        "McDonald's Restaurant",
        "2024-03-15",
        "TAX 8.25%",
        "VISA **** 1234",
    ]
    print(f"\n  Sample tokenizations:")
    for text in samples:
        enc = tokenizer.encode(text)
        # Show without [CLS]/[SEP]
        tokens = enc.tokens[1:-1]
        print(f"    \"{text}\"")
        print(f"      → {tokens}")

    # --- BIO alignment demo ---
    print(f"\n  BIO alignment demo:")
    demo_words = ["McDonald's", "Restaurant", "TOTAL", "$", "12.50"]
    demo_tags  = ["B-NAME",     "I-NAME",     "O",     "O", "B-AMOUNT"]
    aligned_tokens, aligned_tags = align_tags_to_subwords(tokenizer, demo_words, demo_tags)
    print(f"    Words: {demo_words}")
    print(f"    Tags:  {demo_tags}")
    print(f"    After subword split:")
    print(f"    Subwords: {aligned_tokens}")
    print(f"    Tags:     {aligned_tags}")

    print(f"\n{'='*60}")
    if unk_rate < 1.0:
        print(f"  ✅ Tokenizer looks good! UNK rate {unk_rate:.2f}% (target: <1%)")
    else:
        print(f"  ⚠️  UNK rate {unk_rate:.2f}% is above 1% — consider increasing vocab_size")
    print(f"{'='*60}\n")


def align_tags_to_subwords(
    tokenizer: Tokenizer,
    words: list[str],
    tags: list[str],
) -> tuple[list[str], list[str]]:
    """
    Align BIO tags to subword tokens after BPE tokenization.

    Rule: if a word with tag B-X is split into N subwords,
          the first subword gets B-X, the rest get I-X.
          If the word's tag is O, all subwords get O.

    This function will be reused in the training pipeline (Phase 5).

    Args:
        tokenizer: trained BPE tokenizer
        words: original word-level tokens
        tags:  BIO tags aligned to words (same length)

    Returns:
        (subword_tokens, subword_tags) — both lists, same length
    """
    assert len(words) == len(tags), f"Length mismatch: {len(words)} words vs {len(tags)} tags"

    sub_tokens = []
    sub_tags   = []

    for word, tag in zip(words, tags):
        # Tokenize the single word (no [CLS]/[SEP] — use encode directly on model)
        encoded = tokenizer.encode(word, add_special_tokens=False)
        pieces = encoded.tokens

        if not pieces:
            # Edge case: empty encoding (shouldn't happen with UNK fallback)
            pieces = ["[UNK]"]

        for i, piece in enumerate(pieces):
            sub_tokens.append(piece)
            if tag == "O":
                sub_tags.append("O")
            elif i == 0:
                # First subword keeps the original tag (B- or I-)
                sub_tags.append(tag)
            else:
                # Subsequent subwords: B-X → I-X, I-X stays I-X
                if tag.startswith("B-"):
                    sub_tags.append("I-" + tag[2:])
                else:
                    sub_tags.append(tag)  # already I-X

    return sub_tokens, sub_tags


def main():
    if not TRAIN_JSONL.exists():
        print(f"❌ {TRAIN_JSONL} not found. Run Phase 2 first.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Phase 3 — BPE Tokenizer Training")
    print(f"{'='*60}")
    print(f"  Source:     {TRAIN_JSONL}")
    print(f"  Vocab size: {VOCAB_SIZE}")
    print(f"  Output:     {OUTPUT_DIR}/")

    # Step 1: Extract corpus
    print(f"\n  Step 1: Extracting corpus from train.jsonl...")
    corpus = extract_corpus(TRAIN_JSONL)
    print(f"    {len(corpus)} documents loaded")

    # Step 2: Train BPE
    print(f"\n  Step 2: Training BPE tokenizer...")
    tokenizer = train_bpe(corpus, OUTPUT_DIR)

    # Step 3: Verify
    verify_tokenizer(tokenizer, corpus)


if __name__ == "__main__":
    main()