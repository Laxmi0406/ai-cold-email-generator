# AI Resume Analyzer & Job Match Assistant

A Streamlit application that compares a PDF resume with a job description, surfaces detected skill gaps, and uses a Groq-hosted LLM to generate actionable resume feedback and interview practice questions.

## Features

- Upload a text-based PDF resume and extract its text.
- Detect technical and transferable skills using a transparent skill dictionary.
- Calculate a keyword match percentage and show matched/missing skills.
- Calculate TF-IDF cosine similarity as a separate, broader text signal.
- Generate AI-powered improvement suggestions, evidence-based bullet rewrites, learning priorities, and role-specific interview questions.
- Download the improvement plan as Markdown.
- Keep API credentials outside source control using environment variables.

## Tech stack

Python · Streamlit · Groq API · Llama 3.3 · PyPDF2 · scikit-learn · TF-IDF

## Run locally

1. Clone this repository.
2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and add your Groq API key:

   ```env
   GROQ_API_KEY=your_actual_key
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

You can also enter the API key in the app at runtime. It is not written to disk.

## How matching works

- **Keyword match:** detected skills in the job description that also appear in the resume divided by all detected job-description skills.
- **TF-IDF similarity:** cosine similarity between resume and job-description text vectors.

These are explainable indicators, not a proprietary ATS score, hiring probability, or guarantee of selection. A skill not detected by the dictionary may still be present in the document.

## Security and privacy

- Do not commit `.env`, API keys, or personal resumes.
- Resume text is processed in the current Streamlit session and sent to Groq only when the user explicitly requests AI feedback.
- Review the provider's current data-handling terms before using sensitive documents.

## Limitations and next steps

- Scanned/image-only PDFs need OCR; this MVP expects selectable PDF text.
- Skill detection uses a curated dictionary and can miss synonyms or context.
- Future improvements: OCR support, editable skill taxonomy, persistent comparison history with user consent, unit tests, and deployment.

## Responsible use

AI suggestions can be inaccurate. Verify every recommendation and never add skills, experience, or achievements you cannot substantiate.


## Run tests

Install the development dependencies and run the test suite:

```bash
pip install -r requirements-dev.txt
pytest -q
```

GitHub Actions runs these tests automatically on pushes and pull requests to `main`.
