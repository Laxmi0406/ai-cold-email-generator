import os
import re
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from analyzer import analyze_match, generate_ai_feedback

load_dotenv()

st.set_page_config(page_title="AI Resume & Job Match Assistant", page_icon="🎯", layout="wide")

st.title("🎯 AI Resume & Job Match Assistant")
st.caption("Understand your resume, compare it with a job description, and prepare a focused improvement plan.")

with st.sidebar:
    st.header("About this app")
    st.write("Upload a PDF resume and paste a job description to get an explainable skills match and AI-powered feedback.")
    st.info("Your resume is processed in memory for this session. Add your Groq API key to enable AI feedback.")

def extract_pdf_text(uploaded_file):
    try:
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
        if not text:
            raise ValueError("No selectable text was found. Please upload a text-based PDF.")
        return text
    except Exception as exc:
        raise ValueError(f"Could not read this PDF: {exc}") from exc

resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste the job description", height=240, placeholder="Paste the complete job description here...")

if st.button("Analyze my match", type="primary", use_container_width=True):
    if not resume_file:
        st.error("Please upload your resume PDF.")
    elif len(job_description.strip()) < 80:
        st.error("Please paste a more complete job description (at least 80 characters).")
    else:
        try:
            resume_text = extract_pdf_text(resume_file)
            result = analyze_match(resume_text, job_description)
            st.session_state["resume_text"] = resume_text
            st.session_state["job_description"] = job_description
            st.session_state["analysis"] = result
        except ValueError as exc:
            st.error(str(exc))

if "analysis" in st.session_state:
    result = st.session_state["analysis"]
    st.divider()
    st.subheader("Your job-match overview")
    score_col, matched_col, missing_col = st.columns(3)
    score_col.metric("Keyword match", f'{result["score"]}%')
    matched_col.metric("Matched skills", len(result["matched"]))
    missing_col.metric("Skills to strengthen", len(result["missing"]))

    left, right = st.columns(2)
    with left:
        st.markdown("#### Skills found in your resume")
        if result["matched"]:
            st.success(" • ".join(result["matched"]))
        else:
            st.write("No direct skill matches found.")
    with right:
        st.markdown("#### Skills mentioned in the job description but not found")
        if result["missing"]:
            st.warning(" • ".join(result["missing"]))
        else:
            st.success("All detected job-description skills were found in your resume.")

    with st.expander("How is the score calculated?"):
        st.write("The score is the percentage of detected skills from the job description that also appear in the resume. It is a transparent keyword-based indicator, not an employer ATS score or a prediction of selection.")

    st.divider()
    st.subheader("AI-powered improvement plan")
    api_key = st.text_input("Groq API key (or set GROQ_API_KEY in .env)", type="password", help="Your key is not saved by this app.")
    if st.button("Generate AI feedback", use_container_width=True):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            st.error("Add a Groq API key to generate AI feedback.")
        else:
            with st.spinner("Reviewing your resume against the role..."):
                try:
                    feedback = generate_ai_feedback(
                        st.session_state["resume_text"],
                        st.session_state["job_description"],
                        result,
                        key,
                    )
                    st.session_state["feedback"] = feedback
                except Exception as exc:
                    st.error(f"AI request failed: {exc}")

    if "feedback" in st.session_state:
        st.markdown(st.session_state["feedback"])
        st.download_button(
            "Download improvement plan",
            data=st.session_state["feedback"],
            file_name="resume_improvement_plan.md",
            mime="text/markdown",
        )

st.divider()
st.caption("Built with Python, Streamlit, Groq LLM, PyPDF2 and scikit-learn. Always review AI suggestions for accuracy before editing your resume.")
