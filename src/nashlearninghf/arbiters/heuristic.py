"""Heuristic arbiter for summary comparison."""

from __future__ import annotations

import re

from .base import ArbiterResult, BaseArbiter


class HeuristicArbiter(BaseArbiter):
    """Fast heuristic arbiter for TL;DR summaries."""

    def __init__(self, margin: float = 2.0):
        self.margin = margin

    def compare(self, prompt: str, response_a: str, response_b: str) -> ArbiterResult:
        score_a = self._score_summary(prompt, response_a)
        score_b = self._score_summary(prompt, response_b)

        diff = score_a - score_b

        if abs(diff) < self.margin:
            winner = "tie"
        elif diff > 0:
            winner = "a"
        else:
            winner = "b"

        return ArbiterResult(
            winner=winner,
            score_a=score_a,
            score_b=score_b,
            reason=f"Heuristic scores: A={score_a:.1f}, B={score_b:.1f}",
            confidence=min(abs(diff) / 10, 1.0),
        )

    def _score_summary(self, original: str, summary: str) -> float:
        if not summary or summary.startswith("[Error"):
            return 0.0

        orig_len = len(original.split())
        summ_len = len(summary.split())

        if summ_len == 0:
            return 0.0

        compression = orig_len / summ_len
        if compression < 3:
            compression_score = compression * 10
        elif compression > 20:
            compression_score = 100 - (compression - 20) * 2
        else:
            compression_score = min(compression * 5, 100)

        orig_words = set(self._extract_keywords(original))
        summ_words = set(self._extract_keywords(summary))

        if orig_words:
            overlap = len(orig_words & summ_words) / len(orig_words)
            overlap_score = overlap * 100
        else:
            overlap_score = 50

        readability_score = self._readability_score(summary)

        total = compression_score * 0.3 + overlap_score * 0.4 + readability_score * 0.3
        return total

    def _extract_keywords(self, text: str) -> list[str]:
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "and",
            "but",
            "if",
            "or",
            "because",
            "until",
            "while",
            "this",
            "that",
            "these",
            "those",
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "you",
            "your",
            "he",
            "him",
            "his",
            "she",
            "her",
            "it",
            "its",
            "they",
            "them",
            "their",
            "what",
            "which",
            "who",
            "whom",
        }

        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return [w for w in words if w not in stop_words]

    def _readability_score(self, text: str) -> float:
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)

        if avg_len < 5:
            len_score = avg_len * 10
        elif avg_len > 30:
            len_score = max(0, 100 - (avg_len - 30) * 5)
        else:
            len_score = 100 - abs(avg_len - 15) * 3

        structure_score = 100 if text[0].isupper() and text[-1] in ".!?" else 50

        return (len_score + structure_score) / 2
