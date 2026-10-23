import os
import sys
import csv
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from matching_engine import match_candidate_to_job, compute_match_batch

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


st.set_page_config(page_title="AI Resume-Job Matching Assistant", layout="wide")
st.title("🧑‍💼 AI-Powered Resume–Job Matching Assistant")
st.caption(
    "Recruitment use case: automatically score a candidate against a job "
    "description, highlight skill gaps, and generate a recruiter-ready "
    "explanation and interview questions using Generative AI."
)

mode = st.radio(
    "Dataset",
    ["Small curated demo (1-on-1 matching)", "Large real-world dataset (1,199 resumes, 157 jobs)"],
    horizontal=True,
)

if mode.startswith("Large"):
    st.subheader("Bulk screening against a real job posting")
    jobs_df = pd.read_csv(os.path.join(DATA_DIR, "job_descriptions_large.csv"))
    resumes_df = pd.read_csv(os.path.join(DATA_DIR, "resumes_large.csv"))

    job_label = st.selectbox(
        "Choose a real job posting",
        [f"{row.job_id} - {row.position} @ {row.company}" for row in jobs_df.itertuples()],
    )
    job_id = job_label.split(" - ")[0]
    job_row = jobs_df[jobs_df["job_id"] == job_id].iloc[0]
    st.text_area("Job Description", job_row["description"], height=150)

    top_n = st.slider("How many top candidates to show", 3, 20, 10)

    if st.button("🔍 Screen all 1,199 resumes", type="primary"):
        with st.spinner(f"Scoring {len(resumes_df):,} resumes against this posting..."):
            results = compute_match_batch(job_row["description"], resumes_df["resume_text"].tolist())
            out = resumes_df.copy()
            out["final_score"] = [r["final_score"] for r in results]
            out["semantic_score"] = [r["semantic_score"] for r in results]
            out["skill_overlap_pct"] = [r["skill_overlap_pct"] for r in results]
            out["matched_skills"] = ["; ".join(r["matched_skills"]) for r in results]
            out = out.sort_values("final_score", ascending=False).head(top_n)

        st.markdown(f"**Top {top_n} candidates out of {len(resumes_df):,} resumes:**")
        st.dataframe(
            out[["resume_id", "category", "final_score", "semantic_score", "skill_overlap_pct", "matched_skills"]],
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Score = 60% TF-IDF semantic similarity + 40% skill-keyword overlap "
            "with the job description."
        )
    st.stop()

jobs = load_csv(os.path.join(DATA_DIR, "job_descriptions.csv"))
resumes = load_csv(os.path.join(DATA_DIR, "resumes.csv"))

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Job Description")
    job_choice = st.selectbox(
        "Pick a sample job, or choose Custom to paste your own",
        ["Custom"] + [f"{j['job_id']} - {j['title']}" for j in jobs],
    )
    if job_choice == "Custom":
        job_title = st.text_input("Job Title", "")
        job_description = st.text_area("Job Description", "", height=200)
    else:
        job = next(j for j in jobs if job_choice.startswith(j["job_id"]))
        job_title = job["title"]
        job_description = st.text_area("Job Description", job["description"], height=200)

with col2:
    st.subheader("2. Candidate Resume")
    resume_choice = st.selectbox(
        "Pick a sample resume, or choose Custom to paste your own",
        ["Custom"] + [f"{r['resume_id']} - {r['candidate_name']}" for r in resumes],
    )
    if resume_choice == "Custom":
        candidate_name = st.text_input("Candidate Name", "")
        resume_text = st.text_area("Resume Text", "", height=200)
    else:
        r = next(r for r in resumes if resume_choice.startswith(r["resume_id"]))
        candidate_name = r["candidate_name"]
        resume_text = st.text_area("Resume Text", r["resume_text"], height=200)

st.divider()

if st.button("🔍 Run AI Match", type="primary"):
    if not job_description or not resume_text:
        st.warning("Please provide both a job description and a resume.")
    else:
        with st.spinner("Scoring candidate and generating explanation..."):
            result = match_candidate_to_job(
                job_title or "Untitled Role", job_description, candidate_name or "Candidate", resume_text
            )

        st.subheader("Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("Overall Match Score", f"{result['final_score']}/100")
        m2.metric("Semantic Similarity", f"{result['semantic_score']}%")
        m3.metric("Skill Overlap", f"{result['skill_overlap_pct']}%")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**✅ Matched Skills**")
            st.write(", ".join(result["matched_skills"]) or "None found")
        with c2:
            st.markdown("**⚠️ Missing Skills**")
            st.write(", ".join(result["missing_skills"]) or "None — full coverage")

        st.markdown("**🤖 AI-Generated Explanation**")
        st.info(result["explanation"])

        if result["interview_questions"]:
            st.markdown("**❓ Suggested Interview Questions**")
            for q in result["interview_questions"]:
                st.write(f"- {q}")

        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.caption(
                "ℹ️ Running in demo mode with rule-based explanations. "
                "Set the ANTHROPIC_API_KEY environment variable to enable "
                "live LLM-generated explanations."
            )
