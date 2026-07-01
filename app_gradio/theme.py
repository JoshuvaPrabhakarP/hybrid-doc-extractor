"""
Custom Gradio theme + CSS matching the portfolio aesthetic:
dark navy background, blue-to-cyan gradient accents, bold white
headlines, pill-shaped badges, rounded gradient buttons.
"""

import gradio as gr


custom_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
).set(
    # Backgrounds — dark navy throughout
    body_background_fill="#0a0e1a",
    body_background_fill_dark="#0a0e1a",
    background_fill_primary="#11162a",
    background_fill_primary_dark="#11162a",
    background_fill_secondary="#161b30",
    background_fill_secondary_dark="#161b30",
    block_background_fill="#11162a",
    block_background_fill_dark="#11162a",
    panel_background_fill="#0d1120",
    panel_background_fill_dark="#0d1120",

    # Borders
    border_color_primary="#23293f",
    border_color_primary_dark="#23293f",
    block_border_color="#23293f",
    block_border_color_dark="#23293f",
    input_border_color="#23293f",
    input_border_color_dark="#23293f",

    # Text
    body_text_color="#e2e6f0",
    body_text_color_dark="#e2e6f0",
    body_text_color_subdued="#9aa3bc",
    body_text_color_subdued_dark="#9aa3bc",
    block_label_text_color="#9aa3bc",
    block_label_text_color_dark="#9aa3bc",
    block_title_text_color="#ffffff",
    block_title_text_color_dark="#ffffff",

    # Buttons — gradient blue-to-cyan, pill-shaped
    button_primary_background_fill="linear-gradient(90deg, #3b82f6, #22d3ee)",
    button_primary_background_fill_dark="linear-gradient(90deg, #3b82f6, #22d3ee)",
    button_primary_background_fill_hover="linear-gradient(90deg, #2563eb, #0891b2)",
    button_primary_background_fill_hover_dark="linear-gradient(90deg, #2563eb, #0891b2)",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_primary_border_color="transparent",
    button_secondary_background_fill="#1a2036",
    button_secondary_background_fill_dark="#1a2036",
    button_secondary_text_color="#e2e6f0",
    button_secondary_text_color_dark="#e2e6f0",
    button_secondary_border_color="#2a3150",

    # Inputs
    input_background_fill="#0d1120",
    input_background_fill_dark="#0d1120",

    # Shadows / radius
    block_radius="16px",
    button_large_radius="999px",
    button_small_radius="999px",
    input_radius="12px",
)


custom_css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

/* Headline styling — bold, white, large like the portfolio hero */
#app-title h1 {
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem !important;
}

#app-title .gradient-word {
    background: linear-gradient(90deg, #3b82f6, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

#app-subtitle {
    color: #9aa3bc !important;
    font-size: 1.05rem !important;
    margin-bottom: 1.2rem !important;
}

/* Pill badge, like "Open to Opportunities" on the portfolio */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(34, 211, 153, 0.1);
    border: 1px solid rgba(34, 211, 153, 0.3);
    color: #4ade80;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 1rem;
}
.status-pill .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #4ade80;
    box-shadow: 0 0 6px #4ade80;
}

/* Field result cards (DATE/AMOUNT/NAME outputs) */
.field-card {
    background: #11162a;
    border: 1px solid #23293f;
    border-radius: 14px;
    padding: 14px 18px;
    text-align: center;
}
.field-card .field-label {
    color: #9aa3bc;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
.field-card .field-value {
    color: #ffffff;
    font-size: 1.15rem;
    font-weight: 600;
}

/* Tab styling */
.tabitem {
    border: none !important;
}
"""
