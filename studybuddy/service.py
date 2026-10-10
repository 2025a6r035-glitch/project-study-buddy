"""StudyBuddy service: one object the Streamlit UI and the FastAPI server both use."""
from __future__ import annotations

import json
import os
from datetime import date

from .graph import build_graph
from .llm import GeminiLLM, make_embedder
from .planner import parse_topics
from .rag import NoteIndex, read_file
from .tracker import mastery_of
from .workspace import Workspace


class StudyBuddy:
    def __init__(self, llm=None, embedder=None, api_key: str | None = None,
                 persist_path: str | None = None, today=None):
        self.llm = llm or GeminiLLM(api_key=api_key)
        self.index = NoteIndex(embedder or make_embedder(api_key))
        self.ws = Workspace()
        self.quizzes: dict[str, dict] = {}
        self.persist_path = persist_path
        self.today = today or date.today
        self.graph = build_graph(self)
        if persist_path and os.path.exists(persist_path):
            try:
                with open(persist_path, encoding="utf-8") as f:
                    self.import_state(f.read())
            except Exception:
                pass  # a corrupt save file should never stop the app starting

    # ---- notes -------------------------------------------------------------
    def add_notes(self, name: str, data: bytes) -> int:
        text = read_file(name, data)
        if not text.strip():
            raise ValueError(f"No readable text found in '{name}'. Scanned PDFs need OCR first.")
        return self.index.add(name, text)

    # ---- the four actions (each runs through the LangGraph) -----------------
    def plan(self, topics_raw: str, exam_date: date, hours_per_day: float) -> list[dict]:
        if hours_per_day <= 0:
            raise ValueError("Hours per day must be greater than zero.")
        self.ws.topics = parse_topics(topics_raw)
        self.ws.exam_date = exam_date.isoformat()
        self.ws.hours_per_day = float(hours_per_day)
        out = self.graph.invoke({"action": "plan"})
        self._save()
        return out["plan"]

    def ask(self, question: str) -> dict:
        return self.graph.invoke({"action": "ask", "question": question})["answer"]

    def make_quiz(self, topic: str, n: int = 5) -> dict:
        return self.graph.invoke({"action": "quiz", "topic": topic, "n": n})["quiz"]

    def submit_quiz(self, quiz_id: str, answers: list) -> dict:
        out = self.graph.invoke({"action": "submit", "quiz_id": quiz_id, "answers": answers})
        self._save()
        return {"grade": out["grade"], "change": out.get("change", ""), "plan": out.get("plan", [])}

    # ---- views / persistence -------------------------------------------------
    def mastery(self) -> dict[str, float]:
        return mastery_of(self.ws.scores)

    def export_state(self) -> str:
        return json.dumps(self.ws.to_dict(), indent=2)

    def import_state(self, text: str) -> None:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("That file is not a StudyBuddy progress file.")
        self.ws = Workspace.from_dict(data)

    def _save(self) -> None:
        if self.persist_path:
            os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as f:
                f.write(self.export_state())
