"""Improved heuristic scoring for TL;DR summaries."""

from __future__ import annotations

import re
from typing import Dict, List


def score_summary(summary: str, prompt: str) -> float:
    """Heuristic quality score for a TL;DR summary."""
    score = 0.0

    post_match = re.search(r"POST:\s*(.*?)\s*TL;DR:", prompt, re.DOTALL)
    post_text = post_match.group(1).strip() if post_match else prompt

    post_length = len(post_text)
    summary_length = len(summary.strip())

    min_ideal = post_length * 0.08
    max_ideal = post_length * 0.25

    if min_ideal <= summary_length <= max_ideal:
        score += 10
    else:
        if summary_length < min_ideal:
            penalty = (min_ideal - summary_length) / min_ideal if min_ideal else 0
            score -= penalty * 8
        else:
            penalty = (summary_length - max_ideal) / max_ideal if max_ideal else 0
            score -= penalty * 5

    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "been",
        "be",
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
        "can",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "my",
        "your",
        "his",
        "her",
        "our",
        "their",
        "this",
        "that",
    }

    post_words = {
        w.lower() for w in post_text.split() if w.lower() not in stop_words
    }
    summary_words = {
        w.lower() for w in summary.split() if w.lower() not in stop_words
    }

    if post_words:
        overlap_ratio = len(post_words & summary_words) / len(post_words)
        score += overlap_ratio * 15

    summary_clean = summary.strip()

    if summary_clean and summary_clean[-1] in ".!?":
        score += 3
    elif summary_clean and summary_clean[-1] in ",;:":
        score -= 2

    if "\n\n" in summary or summary.count("\n") > 2:
        score -= 3

    if "..." in summary or summary.count(".") > 5:
        score -= 2

    if summary_clean and summary_clean[0].isupper():
        score += 1

    summary_words_list = summary.lower().split()
    if summary_words_list:
        unique_ratio = len(set(summary_words_list)) / len(summary_words_list)
        score += unique_ratio * 5

        for i in range(len(summary_words_list) - 1):
            if summary_words_list[i] == summary_words_list[i + 1]:
                score -= 3

    generic_phrases = [
        "need advice",
        "what should i do",
        "help me",
        "not sure what to do",
        "any advice",
        "thoughts?",
        "opinions?",
        "tell me what to do",
    ]

    summary_lower = summary.lower()
    for phrase in generic_phrases:
        if phrase in summary_lower:
            score -= 4

    common_verbs = [
        "is",
        "was",
        "are",
        "were",
        "do",
        "did",
        "have",
        "has",
        "had",
        "want",
        "need",
        "get",
        "got",
        "make",
        "made",
        "think",
        "know",
    ]

    has_verb = any(verb in summary_lower.split() for verb in common_verbs)
    if has_verb:
        score += 3

    return score


def create_better_preference_pairs(
    prompts: List[str],
    generated_responses: Dict[str, List[str]],
    pairs_per_prompt: int = 10,
) -> List[Dict[str, str]]:
    """Create preference pairs using improved heuristic scoring."""
    import random

    preference_pairs = []
    policy_ids = list(generated_responses.keys())

    for idx, prompt in enumerate(prompts):
        for _ in range(pairs_per_prompt):
            p1, p2 = random.sample(policy_ids, 2)

            r1 = generated_responses[p1][idx]
            r2 = generated_responses[p2][idx]

            score1 = score_summary(r1, prompt)
            score2 = score_summary(r2, prompt)

            if score1 > score2:
                chosen, rejected = r1, r2
                score_diff = score1 - score2
            else:
                chosen, rejected = r2, r1
                score_diff = score2 - score1

            if score_diff < 2.0:
                continue

            preference_pairs.append(
                {"prompt": prompt, "chosen": chosen, "rejected": rejected}
            )

    return preference_pairs
