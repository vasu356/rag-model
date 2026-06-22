"""
Unit tests for the RAGLab backend services.

Tests cover:
- Tokenization & numeric detection
- Adaptive top-k logic
- Numeric grounding validation
- Confidence scoring
- Answer cleaning & normalisation
- History serialisation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the backend package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services.retrieval import (
    _adaptive_top_k,
    _build_confidence_score,
    _extract_numeric_tokens,
    _is_numeric_question,
    _tokenize,
    _validate_numeric_grounding,
)
from app.services.answering import (
    _bullets_to_numbered,
    _clean_answer_text,
    _dedupe_and_cap_bullets,
    _dedupe_repeated_sentences,
    _list_to_bullets,
    _normalize_answer_length,
    _normalize_points_format,
    _parse_listish_string,
    _serialize_history,
    _should_expand_answer,
    _wants_numbered_points,
)


# ===================================================================
# Tokenization
# ===================================================================
class TestTokenize:
    def test_basic(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_empty(self):
        assert _tokenize("") == []

    def test_none(self):
        assert _tokenize(None) == []

    def test_punctuation(self):
        assert _tokenize("cost: $1,500") == ["cost", "1", "500"]


# ===================================================================
# Numeric question detection
# ===================================================================
class TestIsNumericQuestion:
    def test_cost(self):
        assert _is_numeric_question("What is the cost?")

    def test_price(self):
        assert _is_numeric_question("What is the price?")

    def test_percent(self):
        assert _is_numeric_question("What percentage?")

    def test_general(self):
        assert not _is_numeric_question("What is RAG?")

    def test_emi(self):
        assert _is_numeric_question("Calculate EMI")


# ===================================================================
# Adaptive top-k
# ===================================================================
class TestAdaptiveTopK:
    def test_numeric_question(self):
        k = _adaptive_top_k("What is the cost?", 4)
        assert 3 <= k <= 6

    def test_long_question(self):
        k = _adaptive_top_k(
            "Can you explain in detail how the retrieval pipeline works?", 4
        )
        assert k >= 5

    def test_short_question(self):
        k = _adaptive_top_k("What is RAG?", 4)
        assert 3 <= k <= 8


# ===================================================================
# Numeric extraction
# ===================================================================
class TestExtractNumericTokens:
    def test_basic(self):
        tokens = _extract_numeric_tokens("The cost is Rs. 1,500 and 20%")
        assert "1,500" in tokens or "1500" in tokens

    def test_no_numbers(self):
        assert _extract_numeric_tokens("Hello world") == set()

    def test_empty(self):
        assert _extract_numeric_tokens("") == set()


# ===================================================================
# Numeric grounding validation
# ===================================================================
class TestValidateNumericGrounding:
    def test_fully_grounded(self):
        result = _validate_numeric_grounding(
            "The cost is 100.",
            ["The cost is 100 dollars."],
        )
        assert result["has_numbers"]
        assert result["grounded_ratio"] == 1.0

    def test_no_numbers(self):
        result = _validate_numeric_grounding(
            "Hello world.",
            ["Some context."],
        )
        assert not result["has_numbers"]
        assert result["grounded_ratio"] == 1.0

    def test_partially_grounded(self):
        result = _validate_numeric_grounding(
            "Cost is 100 and 200.",
            ["Cost is 100."],
        )
        assert result["has_numbers"]
        assert result["grounded_ratio"] < 1.0


# ===================================================================
# Confidence scoring
# ===================================================================
class TestBuildConfidenceScore:
    def test_high_confidence(self):
        score = _build_confidence_score(
            [3.0, 4.0, 5.0],
            {"has_numbers": False, "grounded_ratio": 1.0},
        )
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_low_confidence(self):
        score = _build_confidence_score(
            [-3.0, -2.0],
            {"has_numbers": False, "grounded_ratio": 0.5},
        )
        assert score < 0.5

    def test_empty_scores(self):
        score = _build_confidence_score([], {"has_numbers": False})
        assert score == 0.0


# ===================================================================
# Answer cleaning
# ===================================================================
class TestCleanAnswerText:
    def test_plain_text(self):
        result = _clean_answer_text("Hello world")
        assert result == "Hello world"

    def test_list_input(self):
        result = _clean_answer_text(["Point A", "Point B"])
        assert "- Point A" in result
        assert "- Point B" in result

    def test_json_wrapped(self):
        result = _clean_answer_text(
            json.dumps({"direct_answer": "This is the answer."})
        )
        assert "This is the answer" in result

    def test_empty(self):
        assert _clean_answer_text("") == ""


# ===================================================================
# List / bullet helpers
# ===================================================================
class TestListToBullets:
    def test_basic(self):
        result = _list_to_bullets(["A", "B", "C"])
        assert result == "- A\n- B\n- C"

    def test_empty(self):
        assert _list_to_bullets([]) == ""


class TestDedupeAndCapBullets:
    def test_deduplication(self):
        result = _dedupe_and_cap_bullets("- A\n- B\n- A")
        assert result.count("- A") == 1
        assert result.count("- B") == 1

    def test_cap(self):
        items = "\n".join([f"- Item {i}" for i in range(20)])
        result = _dedupe_and_cap_bullets(items, max_items=5)
        assert result.count("- Item") == 5


class TestBulletsToNumbered:
    def test_conversion(self):
        result = _bullets_to_numbered("- A\n- B\n- C")
        assert "1. A" in result
        assert "2. B" in result
        assert "3. C" in result


class TestParseListishString:
    def test_valid_list(self):
        result = _parse_listish_string('["A", "B", "C"]')
        assert result == ["A", "B", "C"]

    def test_invalid(self):
        assert _parse_listish_string("Hello") is None

    def test_empty(self):
        assert _parse_listish_string("") is None


# ===================================================================
# History serialisation
# ===================================================================
class TestSerializeHistory:
    def test_empty(self):
        assert _serialize_history([]) == ""

    def test_with_turns(self):
        class FakeTurn:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        turns = [
            FakeTurn("user", "Hello"),
            FakeTurn("ai", "Hi there!"),
        ]
        result = _serialize_history(turns)
        assert "User: Hello" in result
        assert "Assistant: Hi there!" in result


# ===================================================================
# Answer length normalisation
# ===================================================================
class TestNormalizeAnswerLength:
    def test_valid(self):
        assert _normalize_answer_length("short") == "short"
        assert _normalize_answer_length("medium") == "medium"
        assert _normalize_answer_length("detailed") == "detailed"

    def test_invalid_defaults_to_medium(self):
        assert _normalize_answer_length("long") == "medium"
        assert _normalize_answer_length("") == "medium"


# ===================================================================
# Should expand answer
# ===================================================================
class TestShouldExpandAnswer:
    def test_short_answer(self):
        assert _should_expand_answer("What is RAG?", "Short.", "medium")

    def test_already_long(self):
        assert not _should_expand_answer(
            "What is RAG?",
            "A" * 200,
            "medium",
        )


# ===================================================================
# Wants numbered points
# ===================================================================
class TestWantsNumberedPoints:
    def test_numbered(self):
        assert _wants_numbered_points("Give me a numbered list")

    def test_not(self):
        assert not _wants_numbered_points("What is RAG?")


# ===================================================================
# Normalize points format
# ===================================================================
class TestNormalizePointsFormat:
    def test_bullets(self):
        result = _normalize_points_format("- A\n- B")
        assert "- A" in result
        assert "- B" in result

    def test_numbered(self):
        result = _normalize_points_format("1. A\n2. B", prefer_numbered=True)
        assert "1. A" in result
        assert "2. B" in result

    def test_empty(self):
        assert _normalize_points_format("") == ""


# ===================================================================
# Dedupe repeated sentences
# ===================================================================
class TestDedupeRepeatedSentences:
    def test_deduplication(self):
        result = _dedupe_repeated_sentences("Hello. Hello. World.")
        assert result == "Hello. World."

    def test_no_duplicates(self):
        result = _dedupe_repeated_sentences("A. B. C.")
        assert result == "A. B. C."