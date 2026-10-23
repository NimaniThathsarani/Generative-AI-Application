# AI-Powered Resume–Job Matching Assistant
### A Generative AI Application for Recruitment

## 1. Business Use Case

Recruiters manually screening dozens or hundreds of resumes against a job description is slow and inconsistent. This project builds a simple Gen AI tool that:

- Scores how well a candidate's resume matches a job description
- Highlights matched skills and skill gaps
- Uses an LLM to generate a plain-language recruiter explanation of the match
- Suggests tailored interview questions based on the candidate's profile

This directly supports the **screening and shortlisting** stage of the recruitment pipeline.

### 1.1 industry evidence

Independent research from McKinsey and Gartner supports resume screening / talent acquisition as one of the highest-value, fastest-growing Gen AI use cases in HR and customer-facing functions:

- **AI adoption in HR is now mainstream, not experimental.** McKinsey's 2025 *State of AI* research found that the large majority of organizations have adopted AI somewhere in the business, and talent acquisition is consistently reported as one of the leading functional use cases, alongside broader enterprise adoption crossing 80%+ for the first time in 2025 [(McKinsey, "The state of AI in 2025")](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai).
- **Generative AI's economic potential is enormous.** McKinsey separately estimates that generative AI could add on the order of $2.6–4.4 trillion in annual economic value across dozens of identified business use cases, spanning functions including HR and customer operations [(McKinsey via Libertify interactive summary)](https://www.libertify.com/interactive-library/state-of-ai-2025-mckinsey-report/).
- **Recruiting specifically is a leading adopter.** Secondary analyses of McKinsey and LinkedIn data report that roughly three-quarters of U.S. organizations were regularly using AI in HR functions by 2025, with talent acquisition frequently cited as the single largest area of AI experimentation inside HR [("AI Adoption in Recruiting: 2025 Year in Review")](https://www.herohunt.ai/blog/ai-adoption-in-recruiting-2025-year-in-review/).
- **The same shift is happening in customer support**, reinforcing that this project's approach (structured scoring + LLM-generated explanation) generalizes beyond recruitment: Gartner surveyed customer service leaders and found the large majority planned to explore or pilot customer-facing conversational generative AI in 2025, up from a small fraction of teams using AI chatbots just a few years earlier [(Gartner, reported via CX Network)](https://www.cxnetwork.com/artificial-intelligence/news/customer-service-leaders-pilot-generative-ai-2025).

**Takeaway for this project:** resume screening sits squarely inside the use case McKinsey and Gartner both flag as high-adoption and high-value — it is repetitive, high-volume, and benefits from an LLM's ability to turn a structured score into a human-readable justification, which is exactly what this tool automates.

## 2. Architecture

The system has two layers:

**Layer 1 — Open-source NLP matching engine (always runs, no API key needed)**
- `TfidfVectorizer` + cosine similarity (scikit-learn) to measure semantic overlap between the job description and resume text
- Keyword-based skill extraction against a recruitment/HR/tech skill taxonomy
- A weighted final score: `0.6 × semantic similarity + 0.4 × skill overlap`

**Layer 2 — Generative AI layer (LLM API)**
- Calls the **Anthropic Claude API** to convert the raw scores into a natural-language explanation and three tailored interview questions
- If no API key is set (`ANTHROPIC_API_KEY` environment variable), the app automatically falls back to a rule-based explanation labeled `[DEMO MODE]`, so the whole pipeline still runs end-to-end without needing a paid key

## 3. Datasets

Two dataset sizes are included:

**Small curated set** (hand-written, for quick 1-on-1 demos)
- `data/job_descriptions.csv` — 5 sample job postings
- `data/resumes.csv` — 7 sample candidate resumes

**Large real-world dataset** (for scale testing / bulk screening)
- `data/resumes_large.csv` — **1,199 real resumes** across **25 job categories** (HR, Engineering, IT, Finance, Healthcare, Sales, etc.). Originally published as the "Resume Dataset" on Kaggle (snehaanbhawal/resume-dataset), scraped from livecareer.com resume examples.
- `data/job_descriptions_large.csv` — **157 real job postings** scraped from Glassdoor, covering IT, Finance, Business Services, Healthcare, Accounting & Legal, and more. Source: `binoydutt/Resume-Job-Description-Matching` (GitHub).

Both were retrieved from their public GitHub mirrors and cleaned (dropped empty/near-empty rows, standardized column names) with the scripts run to produce this submission.
`

## 4. References

1. McKinsey & Company. "The state of AI in 2025: Agents, innovation, and transformation." https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai
2. McKinsey State of AI 2025 findings, summarized. Libertify Interactive Report. https://www.libertify.com/interactive-library/state-of-ai-2025-mckinsey-report/
3. "AI Adoption in Recruiting: 2025 Year in Review" (secondary analysis citing McKinsey and LinkedIn data). HeroHunt.ai. https://www.herohunt.ai/blog/ai-adoption-in-recruiting-2025-year-in-review/
4. Gartner, Inc. "Gartner Survey Reveals 85% of Customer Service Leaders Will Explore or Pilot Customer-Facing Conversational GenAI in 2025." Reported via CX Network. https://www.cxnetwork.com/artificial-intelligence/news/customer-service-leaders-pilot-generative-ai-2025
5. Resume dataset (1,199 resumes, 25 categories) — originally published on Kaggle as "Resume Dataset" (snehaanbhawal), scraped from livecareer.com.
6. Job postings dataset (157 postings) — `binoydutt/Resume-Job-Description-Matching`, GitHub, scraped from Glassdoor.
