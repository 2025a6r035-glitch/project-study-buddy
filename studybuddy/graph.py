"""LangGraph orchestration: the Planner, Tutor and Tracker agents as graph nodes.

    START --(action)--> planner ----------------------> END          (build a plan)
                   |--> tutor_answer -----------------> END          (ask a question)
                   |--> tutor_quiz -------------------> END          (make a quiz)
                   `--> tracker --> planner (re-plan) -> END         (submit a quiz)

The last route is the adaptive loop: quiz results flow from the Tracker back
into the Planner, which reshapes the remaining schedule."""
from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, TypedDict

from . import planner, tracker, tutor


class State(TypedDict, total=False):
    action: str          # "plan" | "ask" | "quiz" | "submit"
    question: str
    topic: str
    n: int
    quiz_id: str
    answers: list
    answer: dict
    quiz: dict
    grade: dict
    change: str
    plan: list


def build_graph(sb: Any):
    """Build the compiled graph. `sb` is the StudyBuddy service holding llm/index/workspace."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("Package 'langgraph' is not installed. Run: pip install langgraph") from exc

    def planner_node(state: State) -> dict:
        ws = sb.ws
        today = sb.today()
        mastery = tracker.mastery_of(ws.scores)

        if state["action"] == "plan":
            if not ws.exam_date:
                raise ValueError("Pick an exam date first.")
            ws.topics = ws.topics or tutor.extract_topics(sb.llm, sb.index)
            ws.weights = planner.estimate_weights(sb.llm, ws.topics)
            ws.plan = planner.build_schedule(ws.topics, ws.weights, today,
                                             date.fromisoformat(ws.exam_date), ws.hours_per_day, mastery)
            return {"plan": ws.plan, "change": ""}

        # action == "submit": re-plan using the new mastery numbers
        grade, topic = state["grade"], state["topic"]
        if not ws.plan or not ws.exam_date:
            return {"plan": ws.plan,
                    "change": f"You scored {round(grade['score'] * 100)}% on {topic}. "
                              "Generate a study plan to see it adapt to your results."}
        before = tracker.remaining_hours(ws.plan, today)
        try:
            ws.plan = planner.replan(ws.plan, ws.topics, ws.weights, today,
                                     date.fromisoformat(ws.exam_date), ws.hours_per_day, mastery)
        except ValueError as exc:
            return {"plan": ws.plan, "change": str(exc)}
        after = tracker.remaining_hours(ws.plan, today)
        return {"plan": ws.plan, "change": tracker.explain_change(before, after, grade, topic)}

    def tutor_answer_node(state: State) -> dict:
        return {"answer": tutor.answer(sb.llm, sb.index, state.get("question", ""))}

    def tutor_quiz_node(state: State) -> dict:
        topic = state.get("topic", "")
        questions = tutor.make_quiz(sb.llm, sb.index, topic, state.get("n", 5))
        quiz = {"id": uuid.uuid4().hex[:8], "topic": topic, "questions": questions}
        sb.quizzes[quiz["id"]] = quiz
        return {"quiz": quiz}

    def tracker_node(state: State) -> dict:
        quiz = sb.quizzes.get(state.get("quiz_id", ""))
        if quiz is None:
            raise KeyError("That quiz was not found. Generate a new one.")
        grade = tutor.grade_quiz(quiz, state.get("answers", []))
        tracker.record_attempt(sb.ws, quiz["topic"], grade, datetime.now().isoformat(timespec="seconds"))
        return {"grade": grade, "topic": quiz["topic"]}

    g = StateGraph(State)
    g.add_node("planner", planner_node)
    g.add_node("tutor_answer", tutor_answer_node)
    g.add_node("tutor_quiz", tutor_quiz_node)
    g.add_node("tracker", tracker_node)
    g.add_conditional_edges(START, lambda s: s["action"], {
        "plan": "planner", "ask": "tutor_answer", "quiz": "tutor_quiz", "submit": "tracker"})
    g.add_edge("tracker", "planner")
    g.add_edge("planner", END)
    g.add_edge("tutor_answer", END)
    g.add_edge("tutor_quiz", END)
    return g.compile()
