import re
from collections import OrderedDict

from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SKILLS = [
    "python", "java", "c++", "c", "javascript", "typescript", "sql", "html", "css",
    "react", "node.js", "express", "mongodb", "postgresql", "mysql", "docker",
    "kubernetes", "aws", "azure", "gcp", "git", "github", "linux", "rest api",
    "machine learning", "deep learning", "natural language processing", "nlp",
    "generative ai", "llm", "langchain", "retrieval augmented generation", "rag",
    "streamlit", "flask", "django", "fastapi", "scikit-learn", "pandas", "numpy",
    "tensorflow", "pytorch", "data structures", "algorithms", "oop", "agile",
    "communication", "leadership", "problem solving", "project management",
]


def _contains_skill(text: str, skill: str) -> bool:
    escaped = re.escape(skill)
    pattern = rf"(?<![a-z0-9+#]){escaped}(?![a-z0-9+#])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def extract_skills(text: str) -> list[str]:
    return [skill for skill in SKILLS if _contains_skill(text, skill)]


def analyze_match(resume_text: str, job_description: str) -> dict:
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)
    matched = [skill for skill in job_skills if skill in resume_skills]
    missing = [skill for skill in job_skills if skill not in resume_skills]
    score = round((len(matched) / len(job_skills)) * 100) if job_skills else 0

    # TF-IDF similarity provides a second, broader text comparison signal.
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    try:
        matrix = vectorizer.fit_transform([resume_text, job_description])
        similarity = round(float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0]) * 100)
    except ValueError:
        similarity = 0

    return {
        "score": score,
        "matched": matched,
        "missing": missing,
        "resume_skills": resume_skills,
        "job_skills": job_skills,
        "text_similarity": similarity,
    }


def generate_ai_feedback(resume_text: str, job_description: str, result: dict, api_key: str) -> str:
    client = Groq(api_key=api_key)
    resume_excerpt = resume_text[:12000]
    job_excerpt = job_description[:8000]
    prompt = f"""
You are a practical early-career resume coach. Compare the candidate's resume with the job description.
Be specific, truthful, concise, and never invent experience, metrics, skills, or achievements.
Return Markdown with these sections:
1. Overall assessment (mention that keyword match is not a hiring probability)
2. Three highest-impact resume improvements
3. Suggested bullet rewrites (only rewrite facts already supported by the resume; mark placeholders clearly)
4. Skills to learn or demonstrate (distinguish missing skills from skills merely not detected)
5. Five role-specific interview questions

Detected matched skills: {", ".join(result["matched"]) or "None"}
Detected missing skills: {", ".join(result["missing"]) or "None"}
Keyword match: {result["score"]}%
TF-IDF text similarity: {result["text_similarity"]}%

RESUME:
{resume_excerpt}

JOB DESCRIPTION:
{job_excerpt}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You give evidence-based career feedback. Do not fabricate candidate credentials."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1800,
    )
    return response.choices[0].message.content or "No feedback was returned."
