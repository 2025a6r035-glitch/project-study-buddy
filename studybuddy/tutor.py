"""Tutor agent: answers from the student's notes (RAG) and writes quizzes."""
from __future__ import annotations

import random

SYSTEM_TUTOR = (
    "You are StudyBuddy, a patient tutor. Answer using ONLY the provided notes. "
    "If the notes do not contain the answer, say so plainly and suggest what to look up. "
    "Keep answers clear and concise."
)

_QUIZ_SCHEMA = (
    'Return JSON exactly like: {"questions": [{"question": "...", "options": ["A", "B", "C", "D"], '
    '"answer_index": 0, "explanation": "..."}]}\n'
    "Rules: exactly 4 distinct options per question; answer_index is the 0-based index of the correct option; "
    "the explanation is one or two sentences; questions must be answerable from the notes."
)


def _context(hits) -> str:
    return "\n\n".join(f"[{i + 1}] ({chunk.source})\n{chunk.text}" for i, (chunk, _) in enumerate(hits))


def answer(llm, index, question: str, k: int = 5) -> dict:
    question = (question or "").strip()
    if not question:
        raise ValueError("Type a question first.")
    hits = index.search(question, k)
    if not hits:
        raise ValueError("Upload some notes first so the tutor has something to read.")
    prompt = (f"NOTES:\n{_context(hits)}\n\nQUESTION: {question}\n\n"
              "Answer using the notes. Cite note numbers like [1] where relevant.")
    text = llm.generate(prompt, system=SYSTEM_TUTOR)
    sources = [{"n": i + 1, "source": c.source, "snippet": c.text[:300]} for i, (c, _) in enumerate(hits)]
    return {"answer": text, "sources": sources}


def _valid_question(q) -> dict | None:
    if not isinstance(q, dict):
        return None
    question = str(q.get("question", "")).strip()
    options = q.get("options")
    if not question or not isinstance(options, list) or len(options) != 4:
        return None
    options = [str(o).strip() for o in options]
    if any(not o for o in options) or len(set(options)) != 4:
        return None
    try:
        idx = int(q.get("answer_index"))
    except (TypeError, ValueError):
        return None
    if not 0 <= idx <= 3:
        return None
    return {"question": question, "options": options, "answer_index": idx,
            "explanation": str(q.get("explanation", "")).strip()}


def make_quiz(llm, index, topic: str, n: int = 5) -> list[dict]:
    topic = (topic or "").strip()
    if not topic:
        raise ValueError("Choose a topic for the quiz.")
    hits = index.search(topic, 6)
    if not hits:
        raise ValueError("Upload some notes first so the quiz can be built from them.")
    prompt = (f'Using ONLY the notes below, write {n} multiple-choice questions about "{topic}".\n'
              f"{_QUIZ_SCHEMA}\n\nNOTES:\n{_context(hits)}")
    data = llm.generate_json(prompt, system="You write fair, clear exam questions.")
    raw = data.get("questions", []) if isinstance(data, dict) else data
    questions = [v for v in (_valid_question(q) for q in (raw or [])) if v]
    if not questions:
        raise ValueError("The model did not produce a usable quiz. Try again.")
    rng = random.Random()
    for q in questions:  # shuffle options so the answer isn't always in the same slot
        order = list(range(4))
        rng.shuffle(order)
        q["options"] = [q["options"][i] for i in order]
        q["answer_index"] = order.index(q["answer_index"])
    return questions[:n]


def grade_quiz(quiz: dict, answers: list) -> dict:
    questions = quiz["questions"]
    if len(answers) != len(questions):
        raise ValueError("Answer every question before submitting.")
    details, correct = [], 0
    for q, a in zip(questions, answers):
        ok = a == q["answer_index"]
        correct += ok
        details.append({
            "question": q["question"],
            "your_answer": q["options"][a] if isinstance(a, int) and 0 <= a < 4 else None,
            "correct_answer": q["options"][q["answer_index"]],
            "is_correct": ok,
            "explanation": q.get("explanation", ""),
        })
    return {"score": correct / len(questions), "correct": correct, "total": len(questions), "details": details}


def extract_topics(llm, index) -> list[str]:
    sample = index.sample_text()
    if not sample:
        raise ValueError("Upload notes or type your topics so a plan can be built.")
    prompt = ('From these study notes, list the 5 to 10 main topics a student should study, as short names.\n'
              'Return JSON like {"topics": ["Topic one", "Topic two"]}.\n\nNOTES:\n' + sample)
    data = llm.generate_json(prompt)
    raw = data.get("topics", []) if isinstance(data, dict) else data
    topics, seen = [], set()
    for t in raw or []:
        t = str(t).strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            topics.append(t)
    if not topics:
        raise ValueError("Couldn't detect topics from the notes. Type them in manually.")
    return topics[:12]
