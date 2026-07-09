"""
Plotly chart builders for the Gradio dashboard — genuinely interactive
(zoom, pan, hover tooltips) compared to the Streamlit/Altair version.
Reads the same evaluation/*.json artifacts from Phase 6.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ARM_ORDER = ["mamba", "hybrid_5_1", "hybrid_4_2", "hybrid_3_3", "hybrid_2_4", "transformer"]
ARM_LABELS = {
    "mamba": "Mamba (0 attn)",
    "hybrid_5_1": "Hybrid 5:1",
    "hybrid_4_2": "Hybrid 4:2",
    "hybrid_3_3": "Hybrid 3:3",
    "hybrid_2_4": "Hybrid 2:4",
    "transformer": "Transformer (6 attn)",
}
LABEL_ORDER = [ARM_LABELS[a] for a in ARM_ORDER]

# Dark theme to match the app
PLOTLY_TEMPLATE = "plotly_dark"
PLOT_BG = "#11162a"
PAPER_BG = "#11162a"
ACCENT_BLUE = "#3b82f6"
ACCENT_CYAN = "#22d3ee"
ACCENT_GREEN = "#34d399"


def _base_layout(fig, height=380):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(family="Inter, sans-serif", color="#e2e6f0"),
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _load_json(name):
    path = PROJECT_ROOT / "evaluation" / f"{name}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def summary_dataframe() -> pd.DataFrame:
    data = _load_json("test_results")
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
            "Test F1": round(s["avg_f1"], 4),
            "Precision": round(s["avg_precision"], 4),
            "Recall": round(s["avg_recall"], 4),
        })
    df = pd.DataFrame(rows).sort_values("Test F1", ascending=False).reset_index(drop=True)
    return df


def ablation_curve_figure() -> go.Figure:
    df = summary_dataframe()
    if df.empty:
        return go.Figure()

    # Re-sort to ablation order for this specific chart
    order_map = {label: i for i, label in enumerate(LABEL_ORDER)}
    df = df.copy()
    df["_order"] = df["Architecture"].map(order_map)
    df = df.sort_values("_order")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Architecture"], y=df["Test F1"],
        mode="lines+markers",
        line=dict(color=ACCENT_BLUE, width=3),
        marker=dict(size=10, color=ACCENT_CYAN),
        hovertemplate="<b>%{x}</b><br>Test F1: %{y:.4f}<extra></extra>",
    ))
    fig.update_yaxes(title="Test F1", range=[0.5, 0.65])
    fig.update_xaxes(tickangle=-25)
    return _base_layout(fig)


def per_field_breakdown_figure() -> go.Figure:
    data = _load_json("test_results")
    if not data:
        return go.Figure()

    per_run = data.get("per_run", [])
    rows = {}
    for r in per_run:
        arm = r["arm"]
        rows.setdefault(arm, {"DATE": [], "AMOUNT": [], "NAME": []})
        rows[arm]["DATE"].append(r["test_DATE_f1"])
        rows[arm]["AMOUNT"].append(r["test_AMOUNT_f1"])
        rows[arm]["NAME"].append(r["test_NAME_f1"])

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

    fig = px.bar(
        df, x="Architecture", y="F1", color="Field", barmode="group",
        category_orders={"Architecture": LABEL_ORDER, "Field": ["DATE", "AMOUNT", "NAME"]},
        color_discrete_map={"DATE": "#fbbf24", "AMOUNT": ACCENT_GREEN, "NAME": ACCENT_BLUE},
    )
    fig.update_xaxes(tickangle=-25)
    return _base_layout(fig, height=420)


def memory_throughput_figures() -> tuple[go.Figure, go.Figure]:
    data = _load_json("benchmark_results")
    if not data:
        return go.Figure(), go.Figure()

    rows = []
    for arm in ARM_ORDER:
        key = f"{arm}_bs16"
        if key not in data:
            continue
        rows.append({
            "Architecture": ARM_LABELS[arm],
            "Memory (MB)": data[key]["peak_gpu_mb"],
            "Throughput (docs/sec)": data[key]["docs_per_sec"],
        })
    df = pd.DataFrame(rows)
    colors = [ACCENT_GREEN if a == "Mamba (0 attn)" else ACCENT_BLUE for a in df["Architecture"]]

    mem_fig = go.Figure(go.Bar(
        x=df["Architecture"], y=df["Memory (MB)"], marker_color=colors,
        hovertemplate="<b>%{x}</b><br>%{y:.1f} MB<extra></extra>",
    ))
    mem_fig.update_layout(title="Peak GPU memory (lower is better)")
    mem_fig.update_xaxes(tickangle=-25)
    _base_layout(mem_fig, height=350)

    tput_fig = go.Figure(go.Bar(
        x=df["Architecture"], y=df["Throughput (docs/sec)"], marker_color=colors,
        hovertemplate="<b>%{x}</b><br>%{y:.1f} docs/sec<extra></extra>",
    ))
    tput_fig.update_layout(title="Throughput (higher is better)")
    tput_fig.update_xaxes(tickangle=-25)
    _base_layout(tput_fig, height=350)

    return mem_fig, tput_fig


def seqlen_scaling_figure() -> go.Figure:
    data = _load_json("seqlen_results")
    if not data:
        return go.Figure()

    rows = []
    for key, val in data.items():
        if val.get("oom") or val["arm"] not in ("mamba", "transformer"):
            continue
        rows.append({
            "Architecture": ARM_LABELS[val["arm"]],
            "Sequence Length": val["seq_len"],
            "Memory (MB)": val["peak_gpu_mb"],
        })
    df = pd.DataFrame(rows)

    fig = px.line(
        df, x="Sequence Length", y="Memory (MB)", color="Architecture",
        markers=True,
        color_discrete_map={"Mamba (0 attn)": ACCENT_GREEN, "Transformer (6 attn)": ACCENT_BLUE},
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=10))
    fig.update_xaxes(type="category")
    return _base_layout(fig)


def seqlen_table_dataframe() -> pd.DataFrame:
    data = _load_json("seqlen_results")
    if not data:
        return pd.DataFrame()

    rows = []
    for key, val in data.items():
        if val.get("oom"):
            rows.append({"Architecture": ARM_LABELS.get(val["arm"], val["arm"]),
                         "Seq Len": val["seq_len"], "Docs/sec": "OOM", "Memory (MB)": "OOM"})
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
    return df


def significance_dataframe() -> pd.DataFrame:
    data = _load_json("significance_results")
    if not data:
        return pd.DataFrame()

    comparisons = data.get("key_comparisons", [])
    rows = []
    for c in comparisons:
        rows.append({
            "Comparison": f"{ARM_LABELS.get(c['arm_a'], c['arm_a'])} vs {ARM_LABELS.get(c['arm_b'], c['arm_b'])}",
            "Δ": round(c["mean_diff"], 4),
            "p-value": round(c["p_value"], 4),
            "Significant?": "✅ Yes" if c["significant_at_05"] else "❌ No",
        })
    return pd.DataFrame(rows)
