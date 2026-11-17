# 🎭 Broadway Show Finder

**A semantic search engine that recommends Broadway shows from a plain-English description.**

Type a story, a mood, or a theme — like *"a love story set in Paris"* or *"a dark story about revenge"* — and the app finds the Broadway shows whose plot synopses are closest in **meaning**, even when they share none of the same words.

🔗 **Live demo:** [huggingface.co/spaces/XY-Wang111/broadway-show-finder](https://huggingface.co/spaces/XY-Wang111/broadway-show-finder)

---

## How It Works

```
Show synopses ──► Embedding model ──► Vectors ──► ChromaDB
                                                     │
User query ────► Embedding model ──► Vector ─────────┘
                                                     │
                                       Most similar shows (cosine similarity)
```

1. **Data:** ~945 Broadway show synopses are loaded and cleaned (missing values and duplicate shows are removed).
2. **Chunking:** Each synopsis is short (about 200 characters), so one synopsis = one chunk.
3. **Embedding:** Every synopsis is converted into a 384-dimensional vector with `all-MiniLM-L6-v2`.
4. **Storage:** The vectors, synopses, and show names are saved in a persistent **ChromaDB** collection that uses cosine similarity.
5. **Search:** The user's query is embedded with the same model, and ChromaDB returns the closest shows with a similarity score.
6. **Interface:** A **Gradio** web app displays the results as ranked cards.

## Dataset

[TidyTuesday 2020-04-28: Broadway Weekly Grosses](https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-04-28) — the `synopses.csv` file, which contains show names and plot synopses scraped from Playbill.

## Tech Stack

| Component | Tool |
|---|---|
| Embedding model | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Vector database | [ChromaDB](https://www.trychroma.com/) |
| Web interface | [Gradio](https://www.gradio.app/) |
| Deployment | [Hugging Face Spaces](https://huggingface.co/spaces) (ZeroGPU) |

## Project Structure

```
├── build_db.ipynb      # Loads and cleans the data, creates embeddings, builds the database
├── chromadb/           # The saved vector database (created by build_db.ipynb)
├── app.py              # Gradio app: loads the database and runs the search
├── requirements.txt    # Python dependencies
└── README.md
```

## Run It Locally

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
pip install gradio ipykernel

# 3. (Optional) Rebuild the database by running build_db.ipynb

# 4. Start the app
python app.py
```

Then open http://127.0.0.1:7860 in your browser.

## Example Queries

- *a love story set in Paris*
- *a funny show about a family*
- *the American revolution and founding fathers*
- *a dark story about revenge*

## Notes

- The model only understands **English**, so queries should be written in English.
- On a ZeroGPU Space, the model is loaded with `device="cpu"`. Searching is fast on CPU, so no GPU quota is used.
- `chromadb` is pinned to a fixed version so that the database built locally can be read on Hugging Face.

---

*Built for AIPI 510: Data Sourcing for Analytics, Duke University (Fall 2026), Week 13 interactive activity.*
