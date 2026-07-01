"""
Hybrid Document Extractor — Gradio + ZeroGPU dashboard.

Run locally:
    cd ~/hybrid-doc-extractor
    python app_gradio/app.py

Deployed on Hugging Face Spaces with ZeroGPU hardware, all 6 architectures
(including CUDA-only Mamba/Hybrid arms) work exactly as they do locally.
"""

import sys
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app_gradio.theme import custom_theme, custom_css
from app_gradio.inference import (
    extract_entities_from_text,
    entities_to_highlighted_format,
    entities_to_field_dict,
    ARM_DISPLAY_NAMES,
)
from app_gradio.charts import (
    ablation_curve_figure,
    per_field_breakdown_figure,
    summary_dataframe,
    memory_throughput_figures,
    seqlen_scaling_figure,
    seqlen_table_dataframe,
    significance_dataframe,
)

# Real test-set receipts (verified to produce sensible, gold-tag-matching
# output — not made-up text) for one-click examples
SAMPLE_RECEIPTS = {
    "New York Bagel (short)": "NewYorkBagel 226IronwoodAve Coeurd'Alene,Idaho83814 Check:8465 Server:KatieB 09/02/12 09:54am 2 1 BAGEL W/LOX&CC 112Oz.Coffee $2.50 $13.40 $1.50 Subtotal: FOODTAX: Subw/Tax Total: Visa $17.40 $1.04 $18.44 $18.44 $18.44 \"Thank-You\"",
    "Panera Bread": "PaneraBread Cafe0994 Dallas,TX75225 Phone: 214-692-1299 AccuracyMartters, Yourordershouldbecorrecteverytime. Ifit'snot,we'llfixitrightaway,and giveyouafreetreatforyourtrouble. Justletanyassociateknow. 4/8/2017 2:23:18PM CheckNumber:250581Cashier:patricia 1YouPick2 1 1 1/2StrwbpoppyChxSa 1/2StkWhteChedPan 5.89 5.89 NoHorseradishSce +SpicyMustard 1 1 1 Soda ChocChipperCookie FrenchBaguette 2.59 0.99 SubTotal Tax Total GiftCard 15.36 1.27 16.63",
    "Sushi Yasuda": "SushiYasuda 204East43rdStreet NewYorkNY10017 (212)972-1001 06/05/13 1:19PM Check3528 Waiter1MGCust2 Table7A 1 1 1 Edamame Kimo AlaCarteSushi Taxable: Sub-total: SalesTax: TotalDue: 9.00 10.50 243.00 262.50 262.50 23.30 285.80 FollowingthecustominJapan SushiYasuda'sservicestaffare fullycompensatedbytheirsalary. Thereforegratuitiesarenotaccepted. Thankyou.",
}

HIGHLIGHT_COLORS = {
    "DATE": "#fbbf24",
    "AMOUNT": "#34d399",
    "NAME": "#60a5fa",
}


def run_extraction(text, arm_choice):
    if not text or not text.strip():
        return [("Enter some text above.", None)], "—", "—", "—"

    results = extract_entities_from_text(text, arm_choice)
    if not results:
        return [("No output.", None)], "—", "—", "—"

    highlighted = entities_to_highlighted_format(results)
    fields = entities_to_field_dict(results)

    return highlighted, fields["DATE"], fields["AMOUNT"], fields["NAME"]


def load_sample(sample_name):
    return SAMPLE_RECEIPTS.get(sample_name, "")


with gr.Blocks(theme=custom_theme, css=custom_css, title="Hybrid Document Extractor") as demo:

    with gr.Column(elem_id="app-title"):
        gr.HTML(
            '<div class="status-pill"><span class="dot"></span> All 6 architectures live on GPU</div>'
        )
        gr.HTML(
            '<h1>Hybrid Document <span class="gradient-word">Extractor</span></h1>'
        )
        gr.HTML(
            '<p id="app-subtitle">Comparing Transformer, Mamba, and Hybrid architectures '
            'for receipt &amp; invoice field extraction - powered by free ZeroGPU.</p>'
        )

    with gr.Tabs():
        # -------------------------------------------------------------
        # TAB 1 — Live Extraction
        # -------------------------------------------------------------
        with gr.TabItem("Live Extraction"):
            with gr.Row():
                with gr.Column(scale=3):
                    text_input = gr.Textbox(
                        label="Paste receipt / invoice text",
                        placeholder="e.g. WALMART SUPERCENTER 2024-03-15 TOTAL $47.82",
                        lines=5,
                    )
                with gr.Column(scale=1):
                    arm_dropdown = gr.Dropdown(
                        choices=[(v, k) for k, v in ARM_DISPLAY_NAMES.items()],
                        value="transformer",
                        label="Architecture",
                    )
                    sample_dropdown = gr.Dropdown(
                        choices=list(SAMPLE_RECEIPTS.keys()),
                        label="Or load a real test-set example",
                        value=None,
                    )
                    extract_btn = gr.Button("Extract fields", variant="primary", size="lg")

            highlighted_output = gr.HighlightedText(
                label="Highlighted output",
                color_map=HIGHLIGHT_COLORS,
                show_legend=True,
            )

            with gr.Row():
                date_card = gr.Textbox(label="DATE", interactive=False)
                amount_card = gr.Textbox(label="AMOUNT", interactive=False)
                name_card = gr.Textbox(label="NAME", interactive=False)

            sample_dropdown.change(load_sample, inputs=sample_dropdown, outputs=text_input)
            extract_btn.click(
                run_extraction,
                inputs=[text_input, arm_dropdown],
                outputs=[highlighted_output, date_card, amount_card, name_card],
            )

        # -------------------------------------------------------------
        # TAB 2 — Architecture Comparison
        # -------------------------------------------------------------
        with gr.TabItem("Architecture Comparison"):
            gr.Markdown("### Hybrid ratio ablation: F1 vs Mamba:Attention mix")
            gr.Plot(ablation_curve_figure())

            gr.Markdown("### Full results table (held-out test set, avg of 3 seeds)")
            gr.DataFrame(summary_dataframe())

            gr.Markdown("### Per-field F1 breakdown")
            gr.Plot(per_field_breakdown_figure())

            gr.Markdown(
                "### Statistical significance (paired t-test, n=3 seeds)\n"
                "⚠️ Only comparisons against Mamba are reliably significant. "
                "Differences among hybrid ratios and pure Transformer are not "
                "statistically distinguishable with this sample size."
            )
            gr.DataFrame(significance_dataframe())

        # -------------------------------------------------------------
        # TAB 3 — Efficiency Benchmarks
        # -------------------------------------------------------------
        with gr.TabItem("Efficiency Benchmarks"):
            gr.Markdown("### Memory & throughput at training scale (batch=16)")
            mem_fig, tput_fig = memory_throughput_figures()
            with gr.Row():
                gr.Plot(mem_fig)
                gr.Plot(tput_fig)

            gr.Markdown("### Sequence length scaling: O(n) vs O(n²)")
            gr.Plot(seqlen_scaling_figure())

            with gr.Accordion("Raw sequence length benchmark numbers", open=False):
                gr.DataFrame(seqlen_table_dataframe())

        # -------------------------------------------------------------
        # TAB 4 — Methodology
        # -------------------------------------------------------------
        with gr.TabItem("Methodology"):
            gr.Markdown("""
## Project overview

**Research question**: How do Transformer, Mamba (state-space), and hybrid
Mamba+Attention architectures compare for structured field extraction (NER)
on receipts and invoices?

This is a controlled comparison — all three architecture families share the
same tokenizer, embedding, and classification head. Only the encoder stack varies.
""")
            with gr.Row():
                gr.Markdown("**Total documents**\n\n# 3,684\n\nWildReceipt + SROIE + CORD")
                gr.Markdown("**Train / Val / Test**\n\n# 2,947 / 368 / 369\n\n80/10/10 split")
                gr.Markdown("**Vocab size**\n\n# 8,000\n\nDomain-specific BPE, 0% UNK")

            gr.Markdown("## Training methodology — 3 rounds of fixes")

            with gr.Accordion("Round 1 — Naive baseline", open=False):
                gr.Markdown("""
LR=3e-4, 20 epochs, dropout=0.1, all-subword tagging, unweighted loss, no CRF.

| Arm | Avg F1 |
|---|---|
| Transformer | 0.066 |
| Hybrid 5:1 | 0.113 |
| **Mamba** | **0.278** ★ winner |

Mamba dominated — sequential bias compensated for poor training setup.
""")

            with gr.Accordion("Round 2 — First-subword pooling only", open=False):
                gr.Markdown("""
Only first subword of each word gets a tag; rest ignored in loss/eval.

| Arm | Avg F1 | Δ vs R1 |
|---|---|---|
| Transformer | 0.033 | -50% |
| Hybrid 5:1 | 0.055 | -51% |
| **Mamba** | **0.277** ★ winner | ~0% |

Pooling alone hurt Transformer/Hybrid — but was a necessary precursor for word-level CRF.
""")

            with gr.Accordion("Round 3 — All fixes combined (final)", open=True):
                gr.Markdown("""
Added: class-weighted loss, CRF layer, LR→1e-4, dropout→0.2, early stopping.

| Arm | Avg Val F1 | Δ vs R1 |
|---|---|---|
| Mamba | 0.531 | +91% |
| Hybrid 5:1 | 0.574 | +408% |
| **Transformer** | **0.619** ★ winner | +838% |

**Ranking completely reversed.** Mamba is robust to poor setups; Transformer
has a much higher ceiling once given proper training tools.
""")

            gr.Markdown("""
## Key findings

1. **Attention presence matters more than exact ratio.** Going from pure Mamba
   to any hybrid gives a large, statistically significant F1 jump. Differences
   *among* hybrid ratios and pure Transformer are not statistically
   distinguishable with n=3 seeds.
2. **Mamba is the clear efficiency winner**: ~2.2× less GPU memory at training
   scale, growing to ~7.5× less at 1024-token sequences (linear vs quadratic scaling).
3. **Mamba's precision is competitive but recall lags**: fewer predicted
   entities overall, but about as accurate when it does predict.
4. **From-scratch training on ~3k documents** produces F1 in the 0.55-0.62
   range — far below fine-tuned BERT-scale models, but enables a fully
   controlled architecture comparison.

## Limitations

- **n=3 seeds** limits statistical power — fine-grained ranking among hybrid
  configs and pure Transformer should be treated as suggestive, not definitive.
- **Synthetic sequence-length benchmark** tests compute/memory scaling only,
  not accuracy at lengths the models weren't trained on.
- **Domain-specific tokenizer and from-scratch training** mean results may
  not transfer directly to fine-tuned, pretrained-backbone settings.

---
📁 Full technical report: `GRAPH_REPORT.md` in the repository
""")


if __name__ == "__main__":
    demo.launch()
