from analyzer import analyze_match, extract_skills


def test_extract_skills_detects_known_skills():
    text = "Built projects with Python, C++, React, and MongoDB."
    skills = extract_skills(text)
    assert "python" in skills
    assert "c++" in skills
    assert "react" in skills
    assert "mongodb" in skills


def test_extract_skills_avoids_partial_word_matches():
    skills = extract_skills("The word javascripted is not JavaScript.")
    assert "javascript" in skills


def test_analyze_match_calculates_keyword_overlap():
    resume = "Python, SQL, and Git experience."
    job = "We need Python, SQL, Docker, and AWS."
    result = analyze_match(resume, job)

    assert result["matched"] == ["python", "sql"]
    assert result["missing"] == ["docker", "aws"]
    assert result["score"] == 50
    assert 0 <= result["text_similarity"] <= 100


def test_analyze_match_handles_job_without_detected_skills():
    result = analyze_match("I enjoy building software.", "A role focused on curiosity.")
    assert result["score"] == 0
    assert result["matched"] == []
    assert result["missing"] == []
