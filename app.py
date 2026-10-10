"""StudyBuddy AI - Streamlit interface.  Run with:  streamlit run app.py"""
import datetime as dt

import pandas as pd
import streamlit as st

from studybuddy import config
from studybuddy.llm import LLMError
from studybuddy.service import StudyBuddy

st.set_page_config(page_title="StudyBuddy AI", page_icon="🎓", layout="wide")
st.title("🎓 StudyBuddy AI")
st.caption("Your notes → an adaptive study plan, a tutor that reads your notes, and quizzes that reshape the plan.")

# ---------------------------------------------------------------- connect ----
if "sb" not in st.session_state:
    key = config.GEMINI_API_KEY or st.session_state.get("api_key", "")
    if not key:
        st.info("Enter your Gemini API key to start (free from Google AI Studio). "
                "To skip this screen, put GEMINI_API_KEY in your .env file.")
        entered = st.text_input("Gemini API key", type="password")
        if st.button("Connect") and entered.strip():
            st.session_state.api_key = entered.strip()
            st.rerun()
        st.stop()
    try:
        st.session_state.sb = StudyBuddy(api_key=key)
    except (LLMError, RuntimeError) as exc:
        st.error(str(exc))
        st.stop()

sb: StudyBuddy = st.session_state.sb
st.session_state.setdefault("processed", set())
st.session_state.setdefault("chat", [])


def run(fn, *args, **kwargs):
    """Call a service method; show a friendly error instead of a traceback."""
    try:
        return fn(*args, **kwargs)
    except (ValueError, KeyError, LLMError) as exc:
        st.error(str(exc.args[0]) if isinstance(exc, KeyError) else str(exc))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Something went wrong: {exc}")
    return None


# ---------------------------------------------------------------- sidebar ----
with st.sidebar:
    st.header("1 · Your notes")
    files = st.file_uploader("Upload PDF, TXT or MD", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.button("Add notes to StudyBuddy", disabled=not files):
        with st.spinner("Reading and indexing your notes..."):
            for f in files:
                marker = (f.name, f.size)
                if marker in st.session_state.processed:
                    continue
                n = run(sb.add_notes, f.name, f.getvalue())
                if n:
                    st.session_state.processed.add(marker)
                    st.success(f"{f.name}: {n} sections indexed")
    if sb.index.warning:
        st.warning(sb.index.warning)
    sources = sb.index.sources()
    if sources:
        st.caption("Indexed: " + ", ".join(f"{k} ({v})" for k, v in sources.items()))

    st.divider()
    st.subheader("Save / load progress")
    st.download_button("Download progress", sb.export_state(), "studybuddy_progress.json", "application/json")
    saved = st.file_uploader("Load a progress file", type=["json"], key="load_progress")
    if st.button("Load progress", disabled=saved is None):
        if run(lambda: (sb.import_state(saved.getvalue().decode("utf-8")), True)[1]):
            st.rerun()
    st.caption("Notes stay in memory only; re-upload them after restarting.")

tab_plan, tab_ask, tab_quiz, tab_progress = st.tabs(["📅 Study plan", "💬 Ask the tutor", "📝 Quiz", "📈 Progress"])

# ------------------------------------------------------------------- plan ----
with tab_plan:
    st.subheader("2 · Build your plan")
    c1, c2 = st.columns([2, 1])
    with c1:
        topics_raw = st.text_area("Syllabus topics (one per line). Leave blank to detect them from your notes.",
                                  value="\n".join(sb.ws.topics), height=150)
    with c2:
        tomorrow = dt.date.today() + dt.timedelta(days=1)
        default_exam = dt.date.fromisoformat(sb.ws.exam_date) if sb.ws.exam_date else dt.date.today() + dt.timedelta(days=30)
        exam = st.date_input("Exam date", value=max(default_exam, tomorrow), min_value=tomorrow)
        hours = st.number_input("Study hours per day", 0.5, 12.0, float(sb.ws.hours_per_day), 0.5)
        if st.button("Generate study plan", type="primary"):
            with st.spinner("The Planner agent is working..."):
                if run(sb.plan, topics_raw, exam, hours) is not None:
                    st.rerun()

    if sb.ws.plan:
        df = pd.DataFrame(sb.ws.plan).rename(columns={"date": "Date", "topic": "Topic", "hours": "Hours", "kind": "Type"})
        today_iso = dt.date.today().isoformat()
        todays = df[df["Date"] == today_iso]
        if not todays.empty:
            st.success("**Today:** " + ", ".join(f"{r.Topic} ({r.Hours}h)" for r in todays.itertuples()))
        left, right = st.columns([3, 2])
        with left:
            st.dataframe(df, hide_index=True)
        with right:
            st.caption("Total hours per topic")
            st.bar_chart(df.groupby("Topic")["Hours"].sum())
    else:
        st.info("No plan yet. Add notes, set your exam date, and click **Generate study plan**.")

# -------------------------------------------------------------------- ask ----
with tab_ask:
    st.subheader("Ask a question about your notes")
    with st.form("ask_form", clear_on_submit=True):
        question = st.text_input("Your question")
        asked = st.form_submit_button("Ask")
    if asked:
        with st.spinner("Searching your notes..."):
            out = run(sb.ask, question)
        if out:
            st.session_state.chat.insert(0, (question, out))
    for q, out in st.session_state.chat:
        st.markdown(f"**You:** {q}")
        st.markdown(out["answer"])
        with st.expander("Sources from your notes"):
            for s in out["sources"]:
                st.markdown(f"**[{s['n']}] {s['source']}**  \n{s['snippet']}...")
        st.divider()

# ------------------------------------------------------------------- quiz ----
with tab_quiz:
    st.subheader("Test yourself")
    if sb.ws.topics:
        topic = st.selectbox("Topic", sb.ws.topics)
    else:
        topic = st.text_input("Topic")
    n_q = st.slider("Number of questions", 3, 8, 5)
    if st.button("Generate quiz"):
        with st.spinner("The Tutor agent is writing your quiz..."):
            quiz = run(sb.make_quiz, topic, n_q)
        if quiz:
            st.session_state.quiz = quiz
            st.session_state.quiz_result = None

    quiz = st.session_state.get("quiz")
    if quiz:
        st.markdown(f"#### Quiz: {quiz['topic']}")
        for i, q in enumerate(quiz["questions"]):
            st.radio(f"{i + 1}. {q['question']}", q["options"], index=None, key=f"q_{quiz['id']}_{i}")
        if st.button("Submit answers", type="primary"):
            answers = []
            for i, q in enumerate(quiz["questions"]):
                picked = st.session_state.get(f"q_{quiz['id']}_{i}")
                answers.append(q["options"].index(picked) if picked is not None else None)
            if None in answers:
                st.warning("Answer every question before submitting.")
            else:
                with st.spinner("The Tracker agent is updating your plan..."):
                    st.session_state.quiz_result = run(sb.submit_quiz, quiz["id"], answers)

        result = st.session_state.get("quiz_result")
        if result:
            grade = result["grade"]
            st.metric("Score", f"{grade['correct']}/{grade['total']}", f"{round(grade['score'] * 100)}%")
            for d in grade["details"]:
                (st.success if d["is_correct"] else st.error)(
                    f"**{d['question']}**  \nYour answer: {d['your_answer']}  \nCorrect: {d['correct_answer']}"
                    + (f"  \n_{d['explanation']}_" if d["explanation"] else ""))
            st.info(result["change"])
            st.caption("Open the Study plan tab to see your updated schedule.")

# --------------------------------------------------------------- progress ----
with tab_progress:
    st.subheader("How you're doing")
    mastery = sb.mastery()
    if mastery:
        st.caption("Average of your last 3 quiz scores per topic (weak topics get more time in your plan).")
        st.bar_chart(pd.Series({t: round(v * 100) for t, v in mastery.items()}, name="Mastery %"))
        weak = [t for t, v in mastery.items() if v < config.MASTERY_THRESHOLD]
        if weak:
            st.warning("Needs more work: " + ", ".join(weak))
        st.dataframe(pd.DataFrame(sb.ws.attempts)[["time", "topic", "correct", "total"]], hide_index=True)
    else:
        st.info("Take a quiz to see your progress here.")
