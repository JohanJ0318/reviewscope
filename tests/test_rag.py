from reviewscope.rag_bedrock import build_prompt, check_citations
from reviewscope.search import Hit


def test_citations_outside_snippets_are_flagged():
    assert check_citations("Runs well [1][2], crashes [7].", 3) == {"cited": [1, 2, 7], "invalid": [7]}


def test_prompt_numbers_snippets_and_marks_verdict():
    hits = [Hit(1, 0.9, 5, "Neon Drift", "fast", 1, 0), Hit(2, 0.8, 5, "Neon Drift", "slow", 0, 0)]
    prompt = build_prompt("How is performance?", hits)
    assert "[1] (recommended) fast" in prompt
    assert "[2] (not recommended) slow" in prompt
    assert prompt.startswith("Game: Neon Drift")
