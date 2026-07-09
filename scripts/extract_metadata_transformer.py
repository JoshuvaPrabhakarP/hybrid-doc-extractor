"""
Step 3b: Transformer-extracted metadata.

Runs the trained Transformer checkpoint (from the extraction project) over
every chunk in the RAG corpus, producing predicted vendor/date/amount
fields — the "real-world, imperfect extraction" track that gets compared
against ground truth in Step 3c's per-field accuracy report.

MUST run in the OLD venv (~/hybrid-doc-extractor/venv) — this imports
model/checkpoint code that only exists there. Writes its output directly
into the NEW repo's data/ directory (a plain file write, not a Python
import) — this is the file-handoff mechanic between the two venvs; the
new venv never needs to know torch/transformers/mamba versions to read
the result.

Usage (from anywhere, with the OLD venv active):
    python extract_metadata_transformer.py \
        --old-repo-root ~/hybrid-doc-extractor \
        --chunks ~/document-intelligence-rag/data/chunks/chunks.jsonl \
        --out ~/document-intelligence-rag/data/metadata/transformer_extracted.jsonl \
        --arm transformer \
        --seed 456
"""

import argparse
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-repo-root", default=os.path.expanduser("~/hybrid-doc-extractor"))
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="transformer")
    ap.add_argument("--seed", type=int, default=456)  # best test F1 per report: 0.6303
    ap.add_argument("--sample-preview", type=int, default=5)
    args = ap.parse_args()

    # Make the old repo's packages importable regardless of cwd — resolved
    # from an explicit path, not a cwd assumption, so this script can be
    # run from anywhere.
    old_repo_root = os.path.abspath(args.old_repo_root)
    sys.path.insert(0, old_repo_root)

    print(f"Importing inference code from {old_repo_root}...")
    try:
        from app_gradio.inference import extract_entities_from_text, entities_to_field_dict
    except ImportError as e:
        print(f"ERROR: could not import app_gradio.inference from {old_repo_root}: {e}")
        print("Make sure this is running in the OLD venv "
              "(source ~/hybrid-doc-extractor/venv/bin/activate) and that "
              "--old-repo-root points at the right directory.")
        return

    print(f"Loading chunks from {args.chunks}...")
    chunks = []
    with open(args.chunks, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"  {len(chunks)} chunks")
    if not chunks:
        print("ERROR: no chunks loaded, nothing to extract.")
        return

    print(f"\nRunning Transformer (arm={args.arm}, seed={args.seed}) over corpus...")
    print("(first call loads the checkpoint from disk — may take a moment)\n")

    records = []
    start = time.time()
    report_every = 200

    for i, chunk in enumerate(chunks):
        results = extract_entities_from_text(chunk["text"], arm_name=args.arm, seed=args.seed)
        fields = entities_to_field_dict(results)

        # entities_to_field_dict uses "—" for absent fields and joins
        # multiple matches with ", " — normalize to match the ground-truth
        # schema's field naming (vendor/date/amount) and None-for-absent
        # convention from chunk_corpus.py's extract_gt_fields().
        def clean(v):
            return None if v == "—" else v

        records.append({
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "predicted": {
                "vendor": clean(fields.get("NAME")),
                "date": clean(fields.get("DATE")),
                "amount": clean(fields.get("AMOUNT")),
            },
        })

        if (i + 1) % report_every == 0 or (i + 1) == len(chunks):
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (len(chunks) - (i + 1)) / rate if rate > 0 else 0
            print(f"  {i+1}/{len(chunks)} ({rate:.1f} chunks/sec, "
                  f"~{remaining:.0f}s remaining)")

    total_elapsed = time.time() - start
    print(f"\nDone in {total_elapsed:.1f}s ({len(records)/total_elapsed:.1f} chunks/sec)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} predictions -> {args.out}")

    # coverage stats, mirroring chunk_corpus.py's ground-truth coverage report
    with_vendor = sum(1 for r in records if r["predicted"]["vendor"])
    with_date = sum(1 for r in records if r["predicted"]["date"])
    with_amount = sum(1 for r in records if r["predicted"]["amount"])
    n = len(records)
    print(f"\nTransformer-predicted field coverage:")
    print(f"  vendor: {with_vendor}/{n} ({with_vendor/n*100:.1f}%)")
    print(f"  date:   {with_date}/{n} ({with_date/n*100:.1f}%)")
    print(f"  amount: {with_amount}/{n} ({with_amount/n*100:.1f}%)")

    if args.sample_preview > 0:
        print(f"\n--- Sample predictions (first {args.sample_preview}) ---")
        for r in records[:args.sample_preview]:
            print(f"\n{r['chunk_id']} ({r['doc_id']}):")
            print(f"  vendor: {r['predicted']['vendor']!r}")
            print(f"  date:   {r['predicted']['date']!r}")
            print(f"  amount: {r['predicted']['amount']!r}")


if __name__ == "__main__":
    main()
