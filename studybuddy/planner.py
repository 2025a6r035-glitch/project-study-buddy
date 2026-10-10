"""Planner agent: turns topics + exam date + daily hours into a day-by-day schedule.

The LLM only estimates how heavy each topic is; the scheduling itself is
deterministic, so the plan always adds up and is never malformed."""
from __future__ import annotations

import re
from datetime import date, timedelta

from . import config


def parse_topics(raw: str) -> list[str]:
    topics, seen = [], set()
    for line in (raw or "").splitlines():
        t = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", line).strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            topics.append(t)
    return topics


def estimate_weights(llm, topics: list[str]) -> dict[str, float]:
    """Ask the LLM how much study time each topic needs (1 light .. 5 heavy)."""
    default = {t: 3.0 for t in topics}
    prompt = ("Rate how much study time each topic typically needs, from 1 (light) to 5 (heavy), "
              "based on difficulty and breadth.\n"
              'Return JSON like {"weights": {"<topic>": 3}} using the exact topic names.\nTopics:\n'
              + "\n".join(f"- {t}" for t in topics))
    try:
        data = llm.generate_json(prompt)
        raw = data.get("weights", {}) if isinstance(data, dict) else {}
    except Exception:
        return default
    out = {}
    for t in topics:
        try:
            out[t] = min(5.0, max(1.0, float(raw.get(t, 3))))
        except (TypeError, ValueError):
            out[t] = 3.0
    return out


def _allocate(topics, weights, mastery, total_hours) -> dict[str, float]:
    """Split total hours by difficulty, boosted for topics the student is weak at."""
    raw = {}
    for t in topics:
        m = mastery.get(t)
        m = 0.5 if m is None else m  # unseen topics are treated as 'medium'
        raw[t] = weights.get(t, 3.0) * (1 + 2 * (1 - m))
    total = sum(raw.values())
    return {t: total_hours * raw[t] / total for t in topics}


def build_schedule(topics, weights, start: date, exam: date, hours_per_day: float,
                   mastery: dict | None = None) -> list[dict]:
    if not topics:
        raise ValueError("No topics to plan. Type your syllabus topics or upload notes.")
    days = (exam - start).days
    if days < 1:
        raise ValueError("The exam date must be after today.")
    if hours_per_day <= 0:
        raise ValueError("Hours per day must be greater than zero.")
    mastery = mastery or {}
    alloc = _allocate(topics, weights, mastery, days * hours_per_day)
    # Weakest topics go first (stable order when nothing is known yet).
    order = sorted(topics, key=lambda t: (mastery.get(t, 0.5), topics.index(t)))
    spans, cursor = [], 0.0
    for t in order:
        spans.append((t, cursor, cursor + alloc[t]))
        cursor += alloc[t]
    sessions = []
    for d in range(days):
        lo, hi = d * hours_per_day, (d + 1) * hours_per_day
        for t, a, b in spans:
            overlap = min(hi, b) - max(lo, a)
            if overlap >= 0.1:
                m = mastery.get(t)
                sessions.append({
                    "date": (start + timedelta(days=d)).isoformat(),
                    "topic": t,
                    "hours": round(overlap, 2),
                    "kind": "review" if m is not None and m < config.MASTERY_THRESHOLD else "study",
                })
    return sessions


def replan(plan, topics, weights, today: date, exam: date, hours_per_day: float, mastery) -> list[dict]:
    """Keep past sessions as they were; rebuild everything from today onward."""
    if today >= exam:
        raise ValueError("The exam date has already arrived, so there is nothing left to re-plan.")
    past = [s for s in plan if s["date"] < today.isoformat()]
    return past + build_schedule(topics, weights, today, exam, hours_per_day, mastery)
