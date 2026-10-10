"""StudyBuddy AI - FastAPI backend.  Run with:  uvicorn api:app --reload
Interactive docs: http://127.0.0.1:8000/docs

Single-user demo server: one shared workspace, progress saved to data/state.json."""
from datetime import date

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from studybuddy.llm import LLMError
from studybuddy.service import StudyBuddy

app = FastAPI(title="StudyBuddy AI API", version="1.0")
_sb: StudyBuddy | None = None


def get_sb() -> StudyBuddy:
    global _sb
    if _sb is None:
        try:
            _sb = StudyBuddy(persist_path="data/state.json")
        except (LLMError, RuntimeError) as exc:
            raise HTTPException(503, str(exc)) from exc
    return _sb


def call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc.args[0])) from exc
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc


class PlanRequest(BaseModel):
    topics: list[str] = Field(default_factory=list, description="Leave empty to detect topics from notes")
    exam_date: date
    hours_per_day: float = 2.0


class AskRequest(BaseModel):
    question: str


class QuizRequest(BaseModel):
    topic: str
    n: int = Field(5, ge=1, le=10)


class SubmitRequest(BaseModel):
    quiz_id: str
    answers: list[int]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/notes")
async def upload_notes(files: list[UploadFile] = File(...)):
    sb = get_sb()
    added = {}
    for f in files:
        added[f.filename] = call(sb.add_notes, f.filename, await f.read())
    return {"sections_indexed": added, "sources": sb.index.sources()}


@app.post("/plan")
def create_plan(req: PlanRequest):
    sb = get_sb()
    return {"plan": call(sb.plan, "\n".join(req.topics), req.exam_date, req.hours_per_day), "topics": sb.ws.topics}


@app.get("/plan")
def read_plan():
    return {"plan": get_sb().ws.plan}


@app.post("/ask")
def ask(req: AskRequest):
    return call(get_sb().ask, req.question)


@app.post("/quiz")
def new_quiz(req: QuizRequest):
    quiz = call(get_sb().make_quiz, req.topic, req.n)
    # Answer keys stay on the server until the quiz is submitted.
    questions = [{"question": q["question"], "options": q["options"]} for q in quiz["questions"]]
    return {"quiz_id": quiz["id"], "topic": quiz["topic"], "questions": questions}


@app.post("/quiz/submit")
def submit_quiz(req: SubmitRequest):
    return call(get_sb().submit_quiz, req.quiz_id, req.answers)


@app.get("/progress")
def progress():
    sb = get_sb()
    return {"mastery": sb.mastery(), "attempts": sb.ws.attempts}
