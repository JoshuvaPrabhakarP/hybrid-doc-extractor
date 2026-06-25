"""
preprocess.py - Convert WildReceipt, CORD, and SROIE into a unified
BIO-tagged format: {"tokens": [...], "tags": [...]}

Label scheme: O, B-DATE, I-DATE, B-AMOUNT, I-AMOUNT, B-NAME, I-NAME
"""

import os
import json
import random
import difflib
from collections import Counter

random.seed(42)

LABELS = ["O", "B-DATE", "I-DATE", "B-AMOUNT", "I-AMOUNT", "B-NAME", "I-NAME"]

WR_LABEL_MAP = {
    1: "NAME",
    7: "DATE",
    17: "AMOUNT",
    19: "AMOUNT",
    23: "AMOUNT",
}

def process_wildreceipt(path="data/raw/wildreceipt/wildreceipt"):
    records = []
    for split_file in ["train.txt", "test.txt"]:
        fpath = os.path.join(path, split_file)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fpath} not found, skipping")
            continue
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                tokens, tags = [], []
                for ann in doc["annotations"]:
                    text = ann["text"].strip()
                    if not text:
                        continue
                    label = ann.get("label", 0)
                    field = WR_LABEL_MAP.get(label)
                    words = text.split()
                    if field:
                        tags.append(f"B-{field}")
                        tags.extend([f"I-{field}"] * (len(words) - 1))
                    else:
                        tags.extend(["O"] * len(words))
                    tokens.extend(words)
                if tokens:
                    records.append({"tokens": tokens, "tags": tags, "source": "wildreceipt"})
    print(f"WildReceipt: {len(records)} documents processed")
    return records


def process_cord():
    from datasets import load_dataset
    records = []
    ds = load_dataset("naver-clova-ix/cord-v2")
    for split in ds:
        for ex in ds[split]:
            try:
                gt = json.loads(ex["ground_truth"])
                gt_parse = gt.get("gt_parse", {})
            except Exception:
                continue

            total_field = gt_parse.get("total", {})
            amount_text = None
            if isinstance(total_field, dict):
                amount_text = total_field.get("total_price")
            elif isinstance(total_field, list) and total_field:
                amount_text = total_field[0].get("total_price")

            if not amount_text:
                continue

            words = str(amount_text).split()
            if not words:
                continue
            tags = ["B-AMOUNT"] + ["I-AMOUNT"] * (len(words) - 1)
            records.append({"tokens": words, "tags": tags, "source": "cord"})

    print(f"CORD: {len(records)} AMOUNT-only examples processed")
    print("  NOTE: CORD examples are short (amount value only) since CORD's")
    print("  ground_truth lacks full receipt text without separate OCR data.")
    return records


def normalize(s):
    return "".join(c.upper() for c in s if c.isalnum() or c.isspace()).strip()

def parse_box_file(path):
    lines = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",", 8)
            if len(parts) < 9:
                continue
            text = parts[8]
            lines.append(text)
    return lines

def fuzzy_best_match(target, candidates, threshold=0.6):
    target_norm = normalize(target)
    if not target_norm:
        return None, 0.0
    best_idx, best_score = None, 0.0
    for i, cand in enumerate(candidates):
        cand_norm = normalize(cand)
        if not cand_norm:
            continue
        if target_norm in cand_norm or cand_norm in target_norm:
            return i, 1.0
        score = difflib.SequenceMatcher(None, target_norm, cand_norm).ratio()
        if score > best_score:
            best_score, best_idx = score, i
    if best_score >= threshold:
        return best_idx, best_score
    return None, best_score

def process_sroie(root="data/raw/sroie/SROIE2019"):
    records = []
    match_stats = Counter()
    field_map = {"company": "NAME", "date": "DATE", "total": "AMOUNT"}

    for split in ["train", "test"]:
        box_dir = os.path.join(root, split, "box")
        ent_dir = os.path.join(root, split, "entities")
        if not os.path.isdir(box_dir):
            continue
        for fname in os.listdir(box_dir):
            box_path = os.path.join(box_dir, fname)
            ent_path = os.path.join(ent_dir, fname)
            if not os.path.exists(ent_path):
                continue

            lines = parse_box_file(box_path)
            try:
                with open(ent_path, encoding="utf-8", errors="ignore") as f:
                    entities = json.load(f)
            except Exception:
                continue

            line_labels = ["O"] * len(lines)
            for key, field in field_map.items():
                value = entities.get(key, "")
                if not value:
                    match_stats[f"{field}_missing"] += 1
                    continue
                idx, score = fuzzy_best_match(value, lines)
                if idx is not None:
                    line_labels[idx] = field
                    match_stats[f"{field}_matched"] += 1
                else:
                    match_stats[f"{field}_unmatched"] += 1

            tokens, tags = [], []
            for text, field in zip(lines, line_labels):
                words = text.split()
                if not words:
                    continue
                if field != "O":
                    tags.append(f"B-{field}")
                    tags.extend([f"I-{field}"] * (len(words) - 1))
                else:
                    tags.extend(["O"] * len(words))
                tokens.extend(words)

            if tokens:
                records.append({"tokens": tokens, "tags": tags, "source": "sroie"})

    print(f"SROIE: {len(records)} documents processed")
    print("  Match stats:", dict(match_stats))
    return records


def main():
    print("=" * 60)
    print("Phase 2: Data Preprocessing")
    print("=" * 60)

    wr_records = process_wildreceipt()
    cord_records = process_cord()
    sroie_records = process_sroie()

    all_records = wr_records + cord_records + sroie_records
    random.shuffle(all_records)

    print(f"\nTotal combined documents: {len(all_records)}")
    source_counts = Counter(r["source"] for r in all_records)
    print("By source:", dict(source_counts))

    n = len(all_records)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)
    splits = {
        "train": all_records[:train_end],
        "val": all_records[train_end:val_end],
        "test": all_records[val_end:],
    }

    os.makedirs("data/processed", exist_ok=True)
    for name, recs in splits.items():
        out_path = f"data/processed/{name}.jsonl"
        with open(out_path, "w") as f:
            for r in recs:
                f.write(json.dumps({"tokens": r["tokens"], "tags": r["tags"]}) + "\n")
        print(f"Wrote {len(recs)} records to {out_path}")

    print("\nPhase 2 preprocessing complete.")


if __name__ == "__main__":
    main()