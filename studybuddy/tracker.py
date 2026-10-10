"""Tracker agent: records quiz results, measures mastery, explains plan changes."""
from __future__ import annotations

from statistics import mean


def mastery_of(scores: dict[str, list[float]], window: int = 3) -> dict[str, float]:
    """Average of the last `window` quiz scores per topic (0..1)."""
    return {t: mean(v[-window:]) for t, v in scores.items() if v}


def record_attempt(ws, topic: str, grade: dict, when: str) -> None:
    ws.scores.setdefault(topic, []).append(grade["score"])
    ws.attempts.append({"time": when, "topic": topic, "score": grade["score"],
                        "correct": grade["correct"], "total": grade["total"]})


def remaining_hours(plan: list[dict], today) -> dict[str, float]:
    out: dict[str, float] = {}
    for s in plan:
        if s["date"] >= today.isoformat():
            out[s["topic"]] = out.get(s["topic"], 0.0) + s["hours"]
    return out


def explain_change(before: dict, after: dict, grade: dict, topic: str) -> str:
    lines = [f"You scored {round(grade['score'] * 100)}% on {topic}."]
    moves = [(t, before.get(t, 0.0), h) for t, h in after.items() if abs(h - before.get(t, 0.0)) >= 0.25]
    if moves:
        lines.append("Your plan was adjusted (hours remaining before the exam):")
        for t, b, a in sorted(moves, key=lambda m: m[2] - m[1], reverse=True):
            lines.append(f"- {t}: {b:.1f}h -> {a:.1f}h")
    else:
        lines.append("Your plan stays about the same.")
    return "\n".join(lines)
