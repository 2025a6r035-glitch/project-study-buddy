# StudyBuddy AI

A personal agentic study planner and tutor. Upload your notes and it will:

1. **Plan** – turn your syllabus + exam date into a day-by-day study schedule (Planner agent)
2. **Teach** – answer questions and write quizzes using *only your own notes* via RAG (Tutor agent)
3. **Adapt** – read your quiz results and re-shape the remaining schedule toward weak topics (Tracker agent)

The three agents are nodes in a **LangGraph** graph. Submitting a quiz runs `Tracker → Planner`, which is the adaptive loop.

```
START ─ plan ───────► Planner ──────────────────────────► END
      ─ ask ────────► Tutor (answer) ───────────────────► END
      ─ quiz ───────► Tutor (quiz) ─────────────────────► END
      ─ submit ─────► Tracker ──► Planner (re-plan) ────► END
```

## Run it (Windows)

```
cd studybuddy
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        (then open .env and paste your Gemini key)
streamlit run app.py
```

Mac/Linux: use `source venv/bin/activate` and `cp .env.example .env`.

Get a free key at <https://aistudio.google.com>. If you skip the `.env` step, the app asks for the key on screen.

**Optional API server** (same logic, as REST endpoints): `uvicorn api:app --reload`, then open <http://127.0.0.1:8000/docs>.

## Demo script (2 minutes)

1. Upload a PDF of your notes in the sidebar → **Add notes**.
2. *Study plan* tab → set the exam date and hours/day → **Generate study plan**.
3. *Ask the tutor* → ask something your notes cover; open **Sources**.
4. *Quiz* → generate a quiz, answer it (get a few wrong on purpose) → **Submit**.
5. Read the "plan was adjusted" message, then open *Study plan* again: the weak topic now has more hours.

## Project layout

```
app.py                 Streamlit UI
api.py                 FastAPI backend (optional)
studybuddy/
  graph.py             LangGraph orchestration (the 3 agents as nodes)
  tutor.py             Tutor agent: RAG answers, quiz writing, grading
  planner.py           Planner agent: LLM difficulty weights + deterministic scheduler
  tracker.py           Tracker agent: mastery scores, plan-change explanation
  rag.py               File reading, chunking, FAISS vector index
  llm.py               Gemini text + embedding clients, offline fallback embedder
  service.py           One class used by both the UI and the API
  workspace.py         Shared state (topics, plan, scores)
tests/test_core.py     Offline tests (no key needed): python tests/test_core.py
```

## How it works (for your viva)

- **RAG:** notes are split into ~250-word overlapping chunks, embedded, and stored in a FAISS cosine-similarity index. The tutor retrieves the top chunks and the LLM answers *only* from them, with source numbers.
- **Planner:** the LLM only rates each topic's difficulty (1–5). The schedule itself is computed by code, so hours always add up and nothing is malformed. Weak topics get up to 3× the weight.
- **Tracker:** mastery = average of the last 3 quiz scores per topic. Topics under 60% are marked `review` and scheduled first.
- **Fallbacks:** if Gemini embeddings fail, the index switches to offline keyword embeddings and shows a warning. Malformed quiz questions from the model are dropped, and answer options are shuffled.

## Deploy (Streamlit Community Cloud)

Push the folder to GitHub (`.env` is git-ignored), create the app at share.streamlit.io pointing at `app.py`, and add `GEMINI_API_KEY = "your_key"` under **Secrets**.

## Troubleshooting

- **Model not found:** model names change. Set `GEN_MODEL` (and `EMBED_MODEL`) in `.env` to the names shown in Google's Gemini API quickstart.
- **Quota / rate-limit errors:** free tiers are limited; wait a minute and retry, or upload smaller notes.
- **"No readable text" for a PDF:** it's probably scanned images; use a text-based PDF.
- **Notes disappear after restart:** by design, notes live in memory. Plans and scores can be saved with *Download progress*.
- **Run the tests:** `python tests/test_core.py` (uses a fake model; needs no internet).
