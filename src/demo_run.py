import csv
import os
import sys

sys.path.append(os.path.dirname(__file__))
from matching_engine import match_candidate_to_job

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    jobs = load_csv(os.path.join(DATA_DIR, "job_descriptions.csv"))
    resumes = load_csv(os.path.join(DATA_DIR, "resumes.csv"))

    # Demo: match every resume against the HR Recruiter role (JD002)
    # and every resume against the Data Analyst role (JD001), to show
    # the model correctly ranks relevant vs irrelevant candidates.
    demo_jobs = [j for j in jobs if j["job_id"] in ("JD001", "JD002")]

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "results.csv")

    rows = []
    for job in demo_jobs:
        print("=" * 70)
        print(f"JOB: {job['title']} ({job['job_id']})")
        print("=" * 70)

        job_results = []
        for r in resumes:
            res = match_candidate_to_job(
                job["title"], job["description"], r["candidate_name"], r["resume_text"]
            )
            job_results.append(res)

        job_results.sort(key=lambda x: x["final_score"], reverse=True)

        for res in job_results:
            print(f"\n Candidate: {res['candidate_name']}")
            print(f" Score: {res['final_score']}/100  "
                  f"(semantic={res['semantic_score']}, skill_overlap={res['skill_overlap_pct']}%)")
            print(f" Matched skills: {', '.join(res['matched_skills']) or '-'}")
            print(f" Missing skills: {', '.join(res['missing_skills']) or '-'}")
            print(f" Explanation: {res['explanation']}")
            if res["interview_questions"]:
                print(" Suggested interview questions:")
                for q in res["interview_questions"]:
                    print(f"   - {q}")

            rows.append({
                "job_id": job["job_id"],
                "job_title": job["title"],
                "candidate_name": res["candidate_name"],
                "final_score": res["final_score"],
                "semantic_score": res["semantic_score"],
                "skill_overlap_pct": res["skill_overlap_pct"],
                "matched_skills": "; ".join(res["matched_skills"]),
                "missing_skills": "; ".join(res["missing_skills"]),
            })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n\nSaved ranked results for all jobs to: {out_path}")


if __name__ == "__main__":
    main()
