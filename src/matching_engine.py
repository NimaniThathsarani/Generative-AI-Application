import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------
# 1. SKILL TAXONOMY (simple keyword list covering recruitment/HR/data/CS roles)
# ---------------------------------------------------------------------
SKILL_KEYWORDS = [
    "python", "sql", "excel", "power bi", "tableau", "machine learning",
    "deep learning", "pandas", "data analysis", "data visualization",
    "communication", "leadership", "recruitment", "talent acquisition",
    "ats", "linkedin recruiter", "interviewing", "onboarding",
    "hr analytics", "employee engagement", "java", "javascript", "react",
    "project management", "negotiation", "sourcing", "screening",
    "payroll", "labor law", "conflict resolution", "customer service",
    "crm", "salesforce", "zendesk", "microsoft office", "presentation",
    "team management", "teamwork", "scheduling", "problem solving",
    "time management", "written communication", "agile", "version control",
]


def extract_skills(text: str) -> set:
    """Extract known skill keywords found in a piece of text."""
    text_low = text.lower()
    return {s for s in SKILL_KEYWORDS if s in text_low}


# ---------------------------------------------------------------------
# 2. OPEN-SOURCE SEMANTIC MATCH SCORE (TF-IDF + cosine similarity)
# ---------------------------------------------------------------------
def compute_match(job_description: str, resume_text: str) -> dict:
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([job_description, resume_text])
    semantic_score = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])

    jd_skills = extract_skills(job_description)
    resume_skills = extract_skills(resume_text)

    matched_skills = sorted(jd_skills & resume_skills)
    missing_skills = sorted(jd_skills - resume_skills)
    extra_skills = sorted(resume_skills - jd_skills)

    skill_overlap_ratio = (
        len(matched_skills) / len(jd_skills) if jd_skills else 0.0
    )

    # Weighted final score: 60% semantic similarity, 40% skill overlap
    final_score = round((0.6 * semantic_score + 0.4 * skill_overlap_ratio) * 100, 1)

    return {
        "final_score": final_score,
        "semantic_score": round(semantic_score * 100, 1),
        "skill_overlap_pct": round(skill_overlap_ratio * 100, 1),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "extra_skills": extra_skills,
    }


def compute_match_batch(job_description: str, resume_texts: list) -> list:
    """
    Efficient one-vs-many scoring: fits a single TF-IDF vectorizer across
    the job description + the whole resume corpus, then scores every
    resume with one vectorized cosine-similarity call. This is what makes
    scoring a large dataset (hundreds/thousands of resumes) practical —
    calling compute_match() in a loop would re-fit a vectorizer per pair.
    """
    vectorizer = TfidfVectorizer(stop_words="english", max_features=8000)
    docs = [job_description] + list(resume_texts)
    tfidf = vectorizer.fit_transform(docs)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    jd_skills = extract_skills(job_description)
    results = []
    for i, text in enumerate(resume_texts):
        resume_skills = extract_skills(text)
        matched_skills = sorted(jd_skills & resume_skills)
        missing_skills = sorted(jd_skills - resume_skills)
        skill_overlap_ratio = (
            len(matched_skills) / len(jd_skills) if jd_skills else 0.0
        )
        final_score = round((0.6 * sims[i] + 0.4 * skill_overlap_ratio) * 100, 1)
        results.append({
            "final_score": final_score,
            "semantic_score": round(sims[i] * 100, 1),
            "skill_overlap_pct": round(skill_overlap_ratio * 100, 1),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
        })
    return results


# ---------------------------------------------------------------------
# 3. GENERATIVE AI LAYER — LLM explanation + interview questions
# ---------------------------------------------------------------------
def _fallback_explanation(job_title, candidate_name, result):
    """Rule-based fallback used when no LLM API key is configured."""
    verdict = (
        "a strong match" if result["final_score"] >= 70 else
        "a moderate match" if result["final_score"] >= 45 else
        "a weak match"
    )
    matched = ", ".join(result["matched_skills"]) or "none of the listed core skills"
    missing = ", ".join(result["missing_skills"]) or "no major gaps"

    explanation = (
        f"[DEMO MODE - rule-based, no LLM key set]\n"
        f"{candidate_name} appears to be {verdict} for the {job_title} role "
        f"(overall score: {result['final_score']}/100). "
        f"They demonstrate: {matched}. "
        f"Areas not clearly shown in the resume: {missing}."
    )

    questions = [
        f"Can you walk us through a project where you used {s}?"
        for s in (result["matched_skills"][:2] or ["your most relevant experience"])
    ]
    if result["missing_skills"]:
        questions.append(
            f"This role also involves {result['missing_skills'][0]} — "
            f"have you had any exposure to that, even informally?"
        )
    return explanation, questions


def generate_llm_explanation(job_title, job_description, candidate_name, resume_text, result):
    """
    Calls the Anthropic Claude API to generate a natural-language
    recruiter explanation + interview questions based on the computed
    match result. Falls back to a rule-based explanation if no API key
    is available, so the demo always produces output.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _fallback_explanation(job_title, candidate_name, result)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        prompt = f"""You are an AI recruiting assistant. A candidate has been automatically
scored against a job description using NLP similarity + skill matching.

Job Title: {job_title}
Job Description: {job_description}

Candidate: {candidate_name}
Resume: {resume_text}

Computed match score: {result['final_score']}/100
Matched skills: {', '.join(result['matched_skills']) or 'none'}
Missing skills: {', '.join(result['missing_skills']) or 'none'}

Write:
1. A short 2-3 sentence recruiter-facing explanation of why this candidate
   is or isn't a good fit, in plain, direct language.
2. Three tailored interview questions a recruiter could ask this candidate,
   as a numbered list.

Keep the whole response under 150 words."""

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in message.content if block.type == "text"
        )
        # crude split: explanation vs questions section
        return text, None
    except Exception as e:
        note, questions = _fallback_explanation(job_title, candidate_name, result)
        return f"[LLM call failed, used fallback: {e}]\n{note}", questions


# ---------------------------------------------------------------------
# 4. Convenience wrapper
# ---------------------------------------------------------------------
def match_candidate_to_job(job_title, job_description, candidate_name, resume_text):
    result = compute_match(job_description, resume_text)
    explanation, questions = generate_llm_explanation(
        job_title, job_description, candidate_name, resume_text, result
    )
    return {
        "candidate_name": candidate_name,
        "job_title": job_title,
        **result,
        "explanation": explanation,
        "interview_questions": questions,
    }
