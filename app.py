import spaces  # Hugging Face ZeroGPU helper (does nothing when run locally)

import html

import chromadb
import gradio as gr
from sentence_transformers import SentenceTransformer

# 1. Load the embedding model (must be the same model used to build the database)
# device="cpu" keeps the model on the CPU. On a ZeroGPU Space, a model placed on
# the (fake) GPU at startup does not work outside @spaces.GPU functions.
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

# 2. Load the saved database
client = chromadb.PersistentClient(path="chromadb")
collection = client.get_collection("broadway")

# ZeroGPU Spaces only start if the app has at least one @spaces.GPU function.
# Our search is fast enough on CPU, so this function is never called
# and visitors' GPU quota is never used.
@spaces.GPU(duration=10)
def gpu_check():
    return "ok"


# Roman numerals for ranking the results
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


# 3. Search function: takes a query and returns the most similar shows as HTML cards
def search(query, n):
    if query.strip() == "":
        return "<p class='hint'>Please describe the kind of show you wish to see.</p>"

    n = int(n)
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n)

    cards = ""
    for i in range(n):
        name = html.escape(results["metadatas"][0][i]["show"])
        text = html.escape(results["documents"][0][i])
        score = 1 - results["distances"][0][i]   # cosine distance -> similarity
        cards += f"""
        <div class="card">
            <div class="card-top">
                <span class="rank">{ROMAN[i]}</span>
                <span class="name">{name}</span>
                <span class="score">{score:.0%} match</span>
            </div>
            <div class="card-line"></div>
            <p class="synopsis">{text}</p>
        </div>
        """
    return cards


# 4. Colors and fonts
GOLD = "#d4af37"
CREAM = "#f4ead5"
WINE = "#1a0509"

theme = gr.themes.Base(
    font=[gr.themes.GoogleFont("EB Garamond"), "Georgia", "serif"],
).set(
    # page background: deep velvet red
    body_background_fill=f"radial-gradient(ellipse at top, #5a0f1e 0%, #2b0710 45%, {WINE} 100%)",
    body_background_fill_dark=f"radial-gradient(ellipse at top, #5a0f1e 0%, #2b0710 45%, {WINE} 100%)",
    body_text_color=CREAM,
    body_text_color_dark=CREAM,
    body_text_color_subdued="#cbb894",
    body_text_color_subdued_dark="#cbb894",
    # boxes around the inputs
    block_background_fill="rgba(20, 4, 8, 0.55)",
    block_background_fill_dark="rgba(20, 4, 8, 0.55)",
    block_border_color="rgba(212, 175, 55, 0.35)",
    block_border_color_dark="rgba(212, 175, 55, 0.35)",
    block_label_text_color=GOLD,
    block_label_text_color_dark=GOLD,
    block_label_background_fill="transparent",
    block_label_background_fill_dark="transparent",
    block_title_text_color=GOLD,
    block_title_text_color_dark=GOLD,
    # text box
    input_background_fill="#12030a",
    input_background_fill_dark="#12030a",
    input_border_color="rgba(212, 175, 55, 0.5)",
    input_border_color_dark="rgba(212, 175, 55, 0.5)",
    input_border_color_focus=GOLD,
    input_border_color_focus_dark=GOLD,
    input_placeholder_color="#8f7a5a",
    input_placeholder_color_dark="#8f7a5a",
    # gold search button
    button_primary_background_fill="linear-gradient(135deg, #f7e08a 0%, #d4af37 50%, #a67c1a 100%)",
    button_primary_background_fill_dark="linear-gradient(135deg, #f7e08a 0%, #d4af37 50%, #a67c1a 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #fff0a8 0%, #e2bf48 50%, #b58b25 100%)",
    button_primary_background_fill_hover_dark="linear-gradient(135deg, #fff0a8 0%, #e2bf48 50%, #b58b25 100%)",
    button_primary_text_color=WINE,
    button_primary_text_color_dark=WINE,
    button_primary_border_color=GOLD,
    button_primary_border_color_dark=GOLD,
    # example buttons
    button_secondary_background_fill="rgba(212, 175, 55, 0.08)",
    button_secondary_background_fill_dark="rgba(212, 175, 55, 0.08)",
    button_secondary_text_color=CREAM,
    button_secondary_text_color_dark=CREAM,
    button_secondary_border_color="rgba(212, 175, 55, 0.4)",
    button_secondary_border_color_dark="rgba(212, 175, 55, 0.4)",
    # slider and other accents
    slider_color=GOLD,
    slider_color_dark=GOLD,
    color_accent=GOLD,
    color_accent_soft="rgba(212, 175, 55, 0.15)",
    color_accent_soft_dark="rgba(212, 175, 55, 0.15)",
    border_color_primary="rgba(212, 175, 55, 0.35)",
    border_color_primary_dark="rgba(212, 175, 55, 0.35)",
    background_fill_primary="#12030a",
    background_fill_primary_dark="#12030a",
    background_fill_secondary="rgba(20, 4, 8, 0.55)",
    background_fill_secondary_dark="rgba(20, 4, 8, 0.55)",
)

# Cinzel: a classic, royal-looking font for titles
head = """
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700&display=swap" rel="stylesheet">
"""

css = """
.gradio-container { max-width: 900px !important; margin: auto; font-size: 18px; }

/* header: pushed down so the page sits closer to the middle */
#header { text-align: center; padding: 11vh 0 18px 0; }
#header .ornament { color: #d4af37; letter-spacing: 0.6em; font-size: 0.95em; }
#header h1 {
    font-family: 'Cinzel', serif; font-weight: 700;
    font-size: 3.1em; letter-spacing: 0.06em; margin: 12px 0 6px 0;
    background: linear-gradient(180deg, #fff3c4 0%, #e6c35c 45%, #b8892a 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    text-shadow: 0 2px 18px rgba(212, 175, 55, 0.25);
}
#header .subtitle { font-style: italic; font-size: 1.25em; color: #e8dcc0; margin: 0; }
#header .divider { color: #d4af37; margin-top: 14px; font-size: 1.3em; }

/* Cinzel for labels and buttons */
#search-btn, #examples *, label span, .label-wrap span {
    font-family: 'Cinzel', serif !important; letter-spacing: 0.03em;
}
textarea, input { font-size: 1.2em !important; font-family: 'EB Garamond', Georgia, serif !important; }
#search-btn { font-size: 1.1em !important; font-weight: 700 !important; min-height: 52px; }

/* result cards */
#results .card {
    position: relative;
    background: linear-gradient(180deg, rgba(40, 8, 16, 0.85), rgba(22, 4, 10, 0.85));
    border: 1px solid rgba(212, 175, 55, 0.55);
    outline: 1px solid rgba(212, 175, 55, 0.2);
    outline-offset: 4px;
    border-radius: 6px;
    padding: 20px 26px;
    margin: 0 6px 22px 6px;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45);
}
#results .card-top { display: flex; align-items: center; gap: 16px; }
#results .rank {
    font-family: 'Cinzel', serif; font-weight: 700; color: #d4af37 !important;
    font-size: 1.3em; min-width: 42px; text-align: center;
    border-right: 1px solid rgba(212, 175, 55, 0.4); padding-right: 14px;
}
#results .name {
    font-family: 'Cinzel', serif; font-weight: 700; color: #f6e7b8 !important;
    font-size: 1.3em; letter-spacing: 0.03em; flex: 1;
}
#results .score {
    font-family: 'Cinzel', serif; font-size: 0.85em; color: #d4af37 !important;
    border: 1px solid #d4af37 !important; padding: 4px 12px; border-radius: 2px;
    white-space: nowrap;
}
#results .card-line {
    height: 1px; margin: 14px 0 12px 0;
    background: linear-gradient(90deg, transparent, rgba(212, 175, 55, 0.6), transparent);
}
#results .synopsis { margin: 0; font-size: 1.12em; line-height: 1.7; color: #f1e6cc; }
.hint { text-align: center; font-style: italic; color: #cbb894; padding: 26px 0; font-size: 1.1em; }

#footer { text-align: center; color: #a8946e; font-size: 0.95em; padding: 18px 0 40px 0; }
#footer .divider { color: #d4af37; display: block; margin-bottom: 8px; }
"""

# 5. Page layout
with gr.Blocks(title="Broadway Show Finder") as demo:
    gr.HTML("""
    <div id="header">
        <div class="ornament">✦ ✦ ✦</div>
        <h1>Broadway Show Finder</h1>
        <p class="subtitle">Describe a story, a mood, or a theme — and discover the shows that match it.</p>
        <div class="divider">❦</div>
    </div>
    """)

    with gr.Row(equal_height=True):
        query_box = gr.Textbox(
            placeholder="e.g. a love story set in Paris",
            show_label=False,
            scale=5,
        )
        search_button = gr.Button("Search", variant="primary", scale=1, elem_id="search-btn")

    n_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Number of results")

    with gr.Column(elem_id="examples"):
        gr.Examples(
            examples=[
                "a love story set in Paris",
                "a funny show about a family",
                "the American revolution and founding fathers",
                "a dark story about revenge",
            ],
            inputs=query_box,
            label="Try an example",
        )

    results_box = gr.HTML("<p class='hint'>Your recommendations will appear here.</p>", elem_id="results")

    gr.HTML("""
    <div id="footer">
        <span class="divider">✦</span>
        Data: TidyTuesday 2020-04-28 Broadway synopses (from Playbill) ·
        Embedding model: all-MiniLM-L6-v2 · Vector DB: ChromaDB
    </div>
    """)

    # Run the search when the button is clicked or Enter is pressed
    search_button.click(fn=search, inputs=[query_box, n_slider], outputs=results_box)
    query_box.submit(fn=search, inputs=[query_box, n_slider], outputs=results_box)

demo.launch(theme=theme, css=css, head=head)
