"""
Streamlit dashboard for hybrid document extractor — Phase 7.

Run with:
    cd ~/hybrid-doc-extractor
    streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.inference import extract_entities_from_text, ARM_DISPLAY_NAMES
from app.charts import (
    render_ablation_curve,
    render_summary_table,
    render_per_field_breakdown,
    render_significance_badges,
    render_memory_throughput_bars,
    render_seqlen_scaling,
    render_seqlen_table,
)

st.set_page_config(
    page_title="Hybrid Document Extractor",
    page_icon="🧾",
    layout="wide",
)

st.title("🧾 Hybrid Document Extractor")
st.caption("Comparing Transformer, Mamba, and Hybrid architectures for receipt/invoice field extraction")

SAMPLE_RECEIPTS = {
    "Simple receipt": "WALMART SUPERCENTER 2024-03-15 TOTAL $47.82",
    "Restaurant bill": "McDonald's Restaurant Date 2024-06-20 Subtotal $12.50 Tax $1.10 Total $13.60",
    "Invoice style": "INVOICE Acme Corp Invoice Date: 2024-01-05 Amount Due: $1,250.00",
}

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Live Extraction",
    "📊 Architecture Comparison",
    "⚡ Efficiency Benchmarks",
    "📋 Methodology",
])

# ---------------------------------------------------------------------------
# TAB 1 — Live Extraction Demo
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Try it yourself")

    col_input, col_settings = st.columns([3, 1])

    with col_settings:
        arm_choice = st.selectbox(
            "Architecture",
            options=list(ARM_DISPLAY_NAMES.keys()),
            format_func=lambda x: ARM_DISPLAY_NAMES[x],
            index=0,
        )
        st.caption("First selection loads the model (~1-2s). Switching back is instant.")

        sample_choice = st.selectbox("Or try a sample:", ["(none)"] + list(SAMPLE_RECEIPTS.keys()))

    with col_input:
        default_text = SAMPLE_RECEIPTS.get(sample_choice, "") if sample_choice != "(none)" else ""
        text_input = st.text_area(
            "Paste receipt/invoice text",
            value=default_text,
            height=120,
            placeholder="e.g. WALMART SUPERCENTER 2024-03-15 TOTAL $47.82",
        )

    run_button = st.button("Extract fields", type="primary")

    if run_button and text_input.strip():
        with st.spinner(f"Running {ARM_DISPLAY_NAMES[arm_choice]}..."):
            try:
                results = extract_entities_from_text(text_input, arm_choice)
            except FileNotFoundError as e:
                st.error(f"Checkpoint not found: {e}")
                results = []

        if results:
            # Color map for tag highlighting
            TAG_COLORS = {
                "DATE": "#fde68a",
                "AMOUNT": "#bbf7d0",
                "NAME": "#bfdbfe",
            }

            # Render highlighted text
            st.markdown("**Highlighted output:**")
            html_parts = []
            for r in results:
                tag = r["tag"]
                word = r["word"]
                if tag == "O":
                    html_parts.append(word)
                else:
                    field = tag.split("-")[-1]
                    color = TAG_COLORS.get(field, "#e5e7eb")
                    html_parts.append(
                        f'<span style="background-color:{color}; padding:2px 4px; '
                        f'border-radius:4px; font-weight:500;" title="{tag}">{word}</span>'
                    )
            st.markdown(" ".join(html_parts), unsafe_allow_html=True)

            # Extracted fields table
            st.markdown("**Extracted fields:**")
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

            cols = st.columns(3)
            for col, field in zip(cols, ["DATE", "AMOUNT", "NAME"]):
                with col:
                    st.metric(field, ", ".join(extracted[field]) if extracted[field] else "—")
        else:
            st.warning("No output — try different text or check the checkpoint exists.")

    elif run_button:
        st.warning("Please enter some text first.")

# ---------------------------------------------------------------------------
# TAB 2-4 — placeholders, built next
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Hybrid ratio ablation: F1 vs Mamba:Attention mix")
    render_ablation_curve()

    st.divider()
    st.subheader("Full results table (held-out test set, avg of 3 seeds)")
    render_summary_table()

    st.divider()
    st.subheader("Per-field F1 breakdown")
    render_per_field_breakdown()

    st.divider()
    st.subheader("Statistical significance (paired t-test, n=3 seeds)")
    st.caption(
        "⚠️ Only comparisons against Mamba are reliably significant. "
        "Differences among hybrid ratios and pure Transformer are not "
        "statistically distinguishable with this sample size."
    )
    render_significance_badges()

with tab3:
    st.subheader("Memory & throughput at training scale (batch=16)")
    render_memory_throughput_bars()

    st.divider()
    st.subheader("Sequence length scaling: O(n) vs O(n²)")
    render_seqlen_scaling()

    with st.expander("Raw sequence length benchmark numbers"):
        render_seqlen_table()

with tab4:
    st.subheader("Project overview")
    st.markdown("""
    **Research question**: How do Transformer, Mamba (state-space), and hybrid
    Mamba+Attention architectures compare for structured field extraction
    (NER) on receipts and invoices?

    This is a controlled comparison — all three architecture families share
    the same tokenizer, embedding, and classification head. Only the encoder
    stack varies.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total documents", "3,684")
        st.caption("WildReceipt (1,739) + SROIE (973) + CORD (972)")
    with col2:
        st.metric("Train / Val / Test", "2,947 / 368 / 369")
        st.caption("80 / 10 / 10 split")
    with col3:
        st.metric("Vocab size", "8,000")
        st.caption("Domain-specific BPE, 0% UNK rate")

    st.divider()
    st.subheader("Training methodology — 3 rounds of fixes")

    with st.expander("Round 1 — Naive baseline", expanded=False):
        st.markdown("""
        LR=3e-4, 20 epochs, dropout=0.1, all-subword tagging, unweighted loss, no CRF.

        | Arm | Avg F1 |
        |---|---|
        | Transformer | 0.066 |
        | Hybrid 5:1 | 0.113 |
        | **Mamba** | **0.278** ★ winner |

        Mamba dominated — sequential bias compensated for poor training setup.
        """)

    with st.expander("Round 2 — First-subword pooling only", expanded=False):
        st.markdown("""
        Only first subword of each word gets a tag; rest ignored in loss/eval.

        | Arm | Avg F1 | Δ vs R1 |
        |---|---|---|
        | Transformer | 0.033 | -50% |
        | Hybrid 5:1 | 0.055 | -51% |
        | **Mamba** | **0.277** ★ winner | ~0% |

        Pooling alone hurt Transformer/Hybrid — but was a necessary precursor for word-level CRF.
        """)

    with st.expander("Round 3 — All fixes combined (final)", expanded=True):
        st.markdown("""
        Added: class-weighted loss, CRF layer, LR→1e-4, dropout→0.2, early stopping.

        | Arm | Avg Val F1 | Δ vs R1 |
        |---|---|---|
        | Mamba | 0.531 | +91% |
        | Hybrid 5:1 | 0.574 | +408% |
        | **Transformer** | **0.619** ★ winner | +838% |

        **Ranking completely reversed.** Mamba is robust to poor setups; Transformer
        has a much higher ceiling once given proper training tools.
        """)

    st.divider()
    st.subheader("Key findings")
    st.markdown("""
    1. **Attention presence matters more than exact ratio.** Going from pure Mamba
       to any hybrid gives a large, statistically significant F1 jump. Differences
       *among* hybrid ratios (5:1 through 2:4) and pure Transformer are not
       statistically distinguishable with n=3 seeds.
    2. **Mamba is the clear efficiency winner**: ~2.2× less GPU memory at training
       scale, and the gap widens to ~3× more at 1024-token sequences due to
       linear vs quadratic attention scaling.
    3. **Mamba's precision is competitive but recall lags**: it predicts fewer
       entities overall, but is about as accurate as Transformer when it does.
    4. **From-scratch training on ~3k documents** produces F1 in the 0.55-0.62
       range — far below fine-tuned BERT-scale models (0.7-0.9), but enables a
       fully controlled architecture comparison without confounding pretrained
       representations.
    """)

    st.divider()
    st.subheader("Limitations")
    st.warning("""
    - **n=3 seeds** limits statistical power — fine-grained ranking among
      hybrid configs and pure Transformer should be treated as suggestive, not definitive.
    - **Synthetic sequence-length benchmark** tests compute/memory scaling only,
      not accuracy at lengths the models weren't trained on.
    - **Domain-specific tokenizer and from-scratch training** mean results may
      not transfer directly to fine-tuned, pretrained-backbone settings.
    """)

    st.divider()
    st.caption("📁 Full technical report: `graphify-out/GRAPH_REPORT.md` in the repository")
    st.caption("🔗 github.com/JoshuvaPrabhakarP/hybrid-doc-extractor")
