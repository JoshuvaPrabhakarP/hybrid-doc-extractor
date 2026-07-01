"""
Chart-building helpers for the dashboard — reads Phase 6 evaluation JSONs
and renders them as Altair charts (gives us explicit control over category
ordering, which st.line_chart/st.bar_chart don't allow — they alphabetize
string axes by default, which scrambles our intended ablation order).
"""

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# This order matters: it's the ablation progression (0 attn -> 6 attn),
# not alphabetical. Every chart below explicitly sorts by this list.
ARM_ORDER = ["mamba", "hybrid_5_1", "hybrid_4_2", "hybrid_3_3", "hybrid_2_4", "transformer"]
ARM_LABELS = {
    "mamba": "Mamba (0 attn)",
    "hybrid_5_1": "Hybrid 5:1",
    "hybrid_4_2": "Hybrid 4:2",
    "hybrid_3_3": "Hybrid 3:3",
    "hybrid_2_4": "Hybrid 2:4",
    "transformer": "Transformer (6 attn)",
}
LABEL_ORDER = [ARM_LABELS[a] for a in ARM_ORDER]  # the sort order to pass to Altair


@st.cache_data
def load_test_results() -> dict:
    path = PROJECT_ROOT / "evaluation" / "test_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_significance_results() -> dict:
    path = PROJECT_ROOT / "evaluation" / "significance_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_benchmark_results() -> dict:
    path = PROJECT_ROOT / "evaluation" / "benchmark_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_seqlen_results() -> dict:
    path = PROJECT_ROOT / "evaluation" / "seqlen_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def build_summary_dataframe() -> pd.DataFrame:
    """One row per arm: avg test F1, precision, recall, params."""
    data = load_test_results()
    if not data:
        return pd.DataFrame()

    summary = data.get("summary", {})
    rows = []
    for arm in ARM_ORDER:
        if arm not in summary:
            continue
        s = summary[arm]
        rows.append({
            "Architecture": ARM_LABELS[arm],
            "_arm": arm,
            "Test F1": s["avg_f1"],
            "Precision": s["avg_precision"],
            "Recall": s["avg_recall"],
        })
    return pd.DataFrame(rows)


def render_ablation_curve():
    """The headline 6-point ablation curve chart, in correct ablation order."""
    df = build_summary_dataframe()
    if df.empty:
        st.warning("No test results found — run `python -m evaluation.evaluate` first.")
        return

    chart = (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("Architecture:N", sort=LABEL_ORDER, title=None,
                    axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("Test F1:Q", scale=alt.Scale(domain=[0.5, 0.65])),
            tooltip=["Architecture", "Test F1"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)

    st.caption(
        "F1 jumps sharply from pure Mamba to any hybrid, plateaus across hybrid ratios, "
        "then jumps again to pure Transformer. See Methodology tab for the statistical "
        "significance caveats on this trend."
    )


def render_summary_table():
    """Sortable table of F1, precision, recall, params per arm."""
    df = build_summary_dataframe()
    if df.empty:
        return

    sort_by_f1 = st.toggle("Sort by F1 (ranked)", value=True,
                           help="Off = show in ablation order (Mamba → Transformer)")

    display_df = df[["Architecture", "Test F1", "Precision", "Recall"]].copy()
    if sort_by_f1:
        display_df = display_df.sort_values("Test F1", ascending=False).reset_index(drop=True)
    else:
        order_map = {label: i for i, label in enumerate(LABEL_ORDER)}
        display_df["_order"] = display_df["Architecture"].map(order_map)
        display_df = display_df.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    st.dataframe(
        display_df.style.format({"Test F1": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}"})
        .background_gradient(subset=["Test F1"], cmap="Greens"),
        use_container_width=True,
        hide_index=True,
    )


def render_per_field_breakdown():
    """Grouped bars: DATE / AMOUNT / NAME F1 per architecture, in ablation order."""
    data = load_test_results()
    if not data:
        return

    per_run = data.get("per_run", [])
    rows = {}
    for r in per_run:
        arm = r["arm"]
        rows.setdefault(arm, {"DATE": [], "AMOUNT": [], "NAME": []})
        rows[arm]["DATE"].append(r["test_DATE_f1"])
        rows[arm]["AMOUNT"].append(r["test_AMOUNT_f1"])
        rows[arm]["NAME"].append(r["test_NAME_f1"])

    # Long-form data for Altair grouped bars (one row per arm x field)
    long_rows = []
    for arm in ARM_ORDER:
        if arm not in rows:
            continue
        for field in ["DATE", "AMOUNT", "NAME"]:
            values = rows[arm][field]
            long_rows.append({
                "Architecture": ARM_LABELS[arm],
                "Field": field,
                "F1": sum(values) / len(values),
            })

    df = pd.DataFrame(long_rows)

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Field:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("F1:Q", scale=alt.Scale(domain=[0, 0.8])),
            color=alt.Color("Field:N", scale=alt.Scale(
                domain=["DATE", "AMOUNT", "NAME"],
                range=["#fbbf24", "#34d399", "#60a5fa"],
            )),
            column=alt.Column("Architecture:N", sort=LABEL_ORDER, title=None,
                               header=alt.Header(labelAngle=-30, labelAlign="right")),
            tooltip=["Architecture", "Field", "F1"],
        )
        .properties(width=100, height=300)
    )
    st.altair_chart(chart, use_container_width=False)


def render_memory_throughput_bars():
    """Side-by-side bars: peak GPU memory and throughput at batch=16."""
    data = load_benchmark_results()
    if not data:
        st.warning("No benchmark results found — run `python -m evaluation.benchmark` first.")
        return

    rows = []
    for arm in ARM_ORDER:
        key = f"{arm}_bs16"
        if key not in data:
            continue
        rows.append({
            "Architecture": ARM_LABELS[arm],
            "Peak GPU Memory (MB)": data[key]["peak_gpu_mb"],
            "Throughput (docs/sec)": data[key]["docs_per_sec"],
        })
    df = pd.DataFrame(rows)

    col1, col2 = st.columns(2)

    with col1:
        st.caption("Peak GPU memory at batch=16 (lower is better)")
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("Architecture:N", sort=LABEL_ORDER, title=None,
                        axis=alt.Axis(labelAngle=-30)),
                y=alt.Y("Peak GPU Memory (MB):Q"),
                color=alt.condition(
                    alt.datum.Architecture == "Mamba (0 attn)",
                    alt.value("#34d399"),
                    alt.value("#60a5fa"),
                ),
                tooltip=["Architecture", "Peak GPU Memory (MB)"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    with col2:
        st.caption("Throughput at batch=16 (higher is better)")
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("Architecture:N", sort=LABEL_ORDER, title=None,
                        axis=alt.Axis(labelAngle=-30)),
                y=alt.Y("Throughput (docs/sec):Q"),
                color=alt.condition(
                    alt.datum.Architecture == "Mamba (0 attn)",
                    alt.value("#34d399"),
                    alt.value("#60a5fa"),
                ),
                tooltip=["Architecture", "Throughput (docs/sec)"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    st.caption(
        "Mamba uses ~2.2× less memory than any attention-containing architecture "
        "(150.8 MB vs 332-339 MB) — the clearest, most reproducible efficiency signal. "
        "Throughput differences are smaller and less consistent across architectures."
    )


def render_seqlen_scaling():
    """Memory growth vs sequence length — the O(n) vs O(n^2) comparison."""
    data = load_seqlen_results()
    if not data:
        st.warning("No sequence length results found — run `python -m evaluation.seqlen_ablation` first.")
        return

    rows = []
    for key, val in data.items():
        if val.get("oom"):
            continue
        rows.append({
            "Architecture": ARM_LABELS.get(val["arm"], val["arm"]),
            "_arm": val["arm"],
            "Sequence Length": val["seq_len"],
            "Peak GPU Memory (MB)": val["peak_gpu_mb"],
            "Throughput (docs/sec)": val["docs_per_sec"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return

    st.caption(
        "Synthetic-input benchmark (not accuracy) — isolates pure compute/memory "
        "scaling behavior as sequence length grows, independent of trained weights."
    )

    # Only show Mamba vs Transformer for the clearest O(n) vs O(n^2) story,
    # with an option to show all 6
    show_all = st.toggle("Show all 6 architectures", value=False,
                         help="Off = just Mamba vs Transformer (the clearest comparison)")

    if not show_all:
        df = df[df["_arm"].isin(["mamba", "transformer"])]

    chart = (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("Sequence Length:O", title="Sequence length (tokens)"),
            y=alt.Y("Peak GPU Memory (MB):Q"),
            color=alt.Color("Architecture:N", sort=LABEL_ORDER),
            tooltip=["Architecture", "Sequence Length", "Peak GPU Memory (MB)"],
        )
        .properties(height=350)
    )
    st.altair_chart(chart, use_container_width=True)

    # Growth ratio callout
    mamba_256 = df[(df["_arm"] == "mamba") & (df["Sequence Length"] == 256)]["Peak GPU Memory (MB)"]
    mamba_1024 = df[(df["_arm"] == "mamba") & (df["Sequence Length"] == 1024)]["Peak GPU Memory (MB)"]
    tf_256 = df[(df["_arm"] == "transformer") & (df["Sequence Length"] == 256)]["Peak GPU Memory (MB)"]
    tf_1024 = df[(df["_arm"] == "transformer") & (df["Sequence Length"] == 1024)]["Peak GPU Memory (MB)"]

    if not (mamba_256.empty or mamba_1024.empty or tf_256.empty or tf_1024.empty):
        mamba_growth = mamba_1024.values[0] / mamba_256.values[0]
        tf_growth = tf_1024.values[0] / tf_256.values[0]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mamba memory growth (256→1024)", f"{mamba_growth:.2f}×",
                      help="Close to linear O(n) scaling")
        with col2:
            st.metric("Transformer memory growth (256→1024)", f"{tf_growth:.2f}×",
                      help="Quadratic O(n²) attention scaling")


def render_seqlen_table():
    """Raw numbers table for the sequence length ablation."""
    data = load_seqlen_results()
    if not data:
        return

    rows = []
    for key, val in data.items():
        if val.get("oom"):
            rows.append({
                "Architecture": ARM_LABELS.get(val["arm"], val["arm"]),
                "Seq Len": val["seq_len"],
                "Docs/sec": "OOM",
                "Memory (MB)": "OOM",
            })
            continue
        rows.append({
            "Architecture": ARM_LABELS.get(val["arm"], val["arm"]),
            "Seq Len": val["seq_len"],
            "Docs/sec": val["docs_per_sec"],
            "Memory (MB)": val["peak_gpu_mb"],
        })

    df = pd.DataFrame(rows)
    order_map = {label: i for i, label in enumerate(LABEL_ORDER)}
    df["_order"] = df["Architecture"].map(order_map)
    df = df.sort_values(["Seq Len", "_order"]).drop(columns="_order").reset_index(drop=True)

    st.dataframe(df, use_container_width=True, hide_index=True)


def render_significance_badges():
    """Show key pairwise significance results as colored badges."""
    data = load_significance_results()
    if not data:
        st.warning("No significance results found.")
        return

    comparisons = data.get("key_comparisons", [])
    if not comparisons:
        return

    st.caption("Paired t-test across 3 seeds — n=3 gives limited statistical power")

    for c in comparisons:
        arm_a = ARM_LABELS.get(c["arm_a"], c["arm_a"])
        arm_b = ARM_LABELS.get(c["arm_b"], c["arm_b"])
        sig = c["significant_at_05"]
        p = c["p_value"]
        diff = c["mean_diff"]

        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"**{arm_a}** vs **{arm_b}**")
        with col2:
            st.write(f"Δ={diff:+.4f}")
        with col3:
            if sig:
                st.success(f"p={p:.4f} ✅")
            else:
                st.info(f"p={p:.4f} ❌")
