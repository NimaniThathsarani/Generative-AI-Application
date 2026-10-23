import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from matching_engine import compute_match_batch, generate_llm_explanation

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def main():
    resumes = pd.read_csv(os.path.join(DATA_DIR, "resumes_large.csv"))
    jobs = pd.read_csv(os.path.join(DATA_DIR, "job_descriptions_large.csv"))

    print(f"Loaded {len(resumes):,} resumes across {resumes['category'].nunique()} categories")
    print(f"Loaded {len(jobs):,} real job postings\n")

    # Pick 5 representative job postings from different industries for the
    # demonstration (scoring against all 157 x 1199 would be overkill for a
    # short demo; the engine scales to that too — this just keeps the
    # printed output readable).
    demo_jobs = (
        jobs[jobs["industry"].isin(
            ["Information Technology", "Finance", "Business Services", "Health Care"]
        )]
        .groupby("industry", group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )
    # Add one more from a different slice for variety
    extra = jobs[~jobs["job_id"].isin(demo_jobs["job_id"])].sample(1, random_state=42)
    demo_jobs = pd.concat([demo_jobs, extra], ignore_index=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    all_rows = []
    summary_rows = []

    for _, job in demo_jobs.iterrows():
        print("=" * 80)
        print(f"JOB: {job['position']}  |  {job['company']}  |  Industry: {job['industry']}")
        print("=" * 80)

        results = compute_match_batch(job["description"], resumes["resume_text"].tolist())
        for res, (_, r) in zip(results, resumes.iterrows()):
            all_rows.append({
                "job_id": job["job_id"],
                "position": job["position"],
                "industry": job["industry"],
                "resume_id": r["resume_id"],
                "resume_category": r["category"],
                "final_score": res["final_score"],
                "semantic_score": res["semantic_score"],
                "skill_overlap_pct": res["skill_overlap_pct"],
                "matched_skills": "; ".join(res["matched_skills"]),
            })

        # rank for this job
        job_df = pd.DataFrame(all_rows)
        job_df = job_df[job_df["job_id"] == job["job_id"]].sort_values(
            "final_score", ascending=False
        )
        top5 = job_df.head(5)
        print("\nTop 5 matching resumes (out of 1,199):")
        print(top5[["resume_id", "resume_category", "final_score", "matched_skills"]]
              .to_string(index=False))

        # Which resume category wins most often in the top 20?
        top20_categories = job_df.head(20)["resume_category"].value_counts()
        print(f"\nMost common resume category in top 20 matches: "
              f"{top20_categories.index[0]} ({top20_categories.iloc[0]}/20)")

        # Generate one LLM/fallback explanation for the #1 candidate
        top_resume = resumes[resumes["resume_id"] == top5.iloc[0]["resume_id"]].iloc[0]
        top_result_full = compute_match_batch(job["description"], [top_resume["resume_text"]])[0]
        explanation, questions = generate_llm_explanation(
            job["position"], job["description"],
            f"Candidate #{top_resume['resume_id']} ({top_resume['category']})",
            top_resume["resume_text"], top_result_full,
        )
        print(f"\nAI explanation for top candidate:\n{explanation}")
        if questions:
            print("Suggested interview questions:")
            for q in questions:
                print(f"  - {q}")
        print()

        summary_rows.append({
            "job_id": job["job_id"],
            "position": job["position"],
            "industry": job["industry"],
            "top_candidate_resume_id": top5.iloc[0]["resume_id"],
            "top_candidate_category": top5.iloc[0]["resume_category"],
            "top_score": top5.iloc[0]["final_score"],
            "dominant_category_in_top20": top20_categories.index[0],
        })

    full_df = pd.DataFrame(all_rows)
    full_df.to_csv(os.path.join(OUT_DIR, "large_scale_results.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_DIR, "large_scale_summary.csv"), index=False)

    print("=" * 80)
    print(f"Scored {len(resumes):,} resumes x {len(demo_jobs)} jobs = {len(full_df):,} total match computations")
    print(f"Full ranked results saved to outputs/large_scale_results.csv")
    print(f"Per-job summary saved to outputs/large_scale_summary.csv")


if __name__ == "__main__":
    main()
