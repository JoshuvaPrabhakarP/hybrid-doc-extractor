#!/usr/bin/env python3
"""
Phase 6, Step 4 — Statistical significance testing.

Tests whether the F1 differences between architectures are statistically
significant, or could plausibly be explained by seed variance alone.

Uses paired t-tests across the 3 seeds (42, 123, 456) — "paired" because
the same 3 seeds were used for every arm, so we're comparing matched
samples rather than independent ones. With only 3 seeds, statistical
power is limited; results should be read as suggestive, not definitive.

Usage:
    cd ~/hybrid-doc-extractor
    python -m evaluation.significance_test
"""

import json
import sys
from itertools import combinations
from pathlib import Path

from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_test_results() -> dict[str, list[float]]:
    """
    Load per-seed test F1 scores for each arm from evaluation/test_results.json.

    Returns:
        dict mapping arm_name -> [f1_seed42, f1_seed123, f1_seed456]
        (ordered by seed to ensure proper pairing across arms)
    """
    results_path = PROJECT_ROOT / "evaluation" / "test_results.json"
    with open(results_path) as f:
        data = json.load(f)

    per_run = data["per_run"]

    # Group by arm, ordered by seed
    arm_seeds = {}
    for r in per_run:
        arm_seeds.setdefault(r["arm"], {})[r["seed"]] = r["test_entity_f1"]

    seed_order = sorted(next(iter(arm_seeds.values())).keys())

    arm_scores = {}
    for arm, seed_dict in arm_seeds.items():
        arm_scores[arm] = [seed_dict[s] for s in seed_order]

    return arm_scores, seed_order


def main():
    arm_scores, seed_order = load_test_results()

    print(f"\n{'='*70}")
    print(f"  Phase 6 — Statistical Significance Testing")
    print(f"{'='*70}")
    print(f"  Method: paired t-test across {len(seed_order)} seeds {seed_order}")
    print(f"  Note: n=3 seeds gives limited statistical power.")
    print(f"        Results below are suggestive, not definitive proof.\n")

    print(f"  Per-arm test F1 by seed:")
    for arm, scores in sorted(arm_scores.items(), key=lambda x: -sum(x[1])/len(x[1])):
        avg = sum(scores) / len(scores)
        print(f"    {arm:<15s}: {[f'{s:.4f}' for s in scores]} → avg={avg:.4f}")

    # Pairwise comparisons — focus on the key research questions
    print(f"\n{'='*70}")
    print(f"  Pairwise paired t-tests (key comparisons)")
    print(f"{'='*70}")

    key_pairs = [
        ("transformer", "mamba"),
        ("transformer", "hybrid_5_1"),
        ("transformer", "hybrid_4_2"),
        ("transformer", "hybrid_3_3"),
        ("transformer", "hybrid_2_4"),
        ("hybrid_5_1", "mamba"),
        ("hybrid_4_2", "hybrid_5_1"),
        ("hybrid_3_3", "hybrid_4_2"),
        ("hybrid_2_4", "hybrid_3_3"),
    ]

    results = []
    for arm_a, arm_b in key_pairs:
        if arm_a not in arm_scores or arm_b not in arm_scores:
            continue

        scores_a = arm_scores[arm_a]
        scores_b = arm_scores[arm_b]

        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

        mean_diff = sum(scores_a)/len(scores_a) - sum(scores_b)/len(scores_b)
        sig = "✅ significant (p<0.05)" if p_value < 0.05 else "❌ not significant (p≥0.05)"

        print(
            f"  {arm_a:<15s} vs {arm_b:<15s} | "
            f"Δ={mean_diff:+.4f} | t={t_stat:+.3f} | p={p_value:.4f} | {sig}"
        )

        results.append({
            "arm_a": arm_a, "arm_b": arm_b,
            "mean_diff": round(mean_diff, 4),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 4),
            "significant_at_05": bool(p_value < 0.05),
        })

    # Full pairwise matrix (all combinations) for completeness
    print(f"\n{'='*70}")
    print(f"  Full pairwise matrix (all {len(arm_scores)} arms)")
    print(f"{'='*70}")

    all_arms = sorted(arm_scores.keys(), key=lambda a: -sum(arm_scores[a])/len(arm_scores[a]))
    full_matrix = {}
    for arm_a, arm_b in combinations(all_arms, 2):
        t_stat, p_value = stats.ttest_rel(arm_scores[arm_a], arm_scores[arm_b])
        full_matrix[f"{arm_a}_vs_{arm_b}"] = {
            "p_value": round(p_value, 4),
            "significant": bool(p_value < 0.05),
        }

    sig_count = sum(1 for v in full_matrix.values() if v["significant"])
    print(f"  {sig_count}/{len(full_matrix)} pairwise comparisons are significant at p<0.05\n")

    for k, v in full_matrix.items():
        marker = "✅" if v["significant"] else "  "
        print(f"    {marker} {k:<35s} p={v['p_value']:.4f}")

    # Save
    output_path = PROJECT_ROOT / "evaluation" / "significance_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "key_comparisons": results,
            "full_matrix": full_matrix,
            "arm_scores": arm_scores,
            "seed_order": seed_order,
            "caveat": "n=3 seeds; results are suggestive given limited statistical power",
        }, f, indent=2)
    print(f"\n  Results saved to: {output_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
