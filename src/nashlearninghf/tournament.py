"""Tournament system for comparing models."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import numpy as np

from .arbiters.base import ArbiterResult, BaseArbiter
from .scoring import score_summary


@dataclass
class MatchResult:
    """Single match result between two models."""

    model_a: str
    model_b: str
    prompt: str
    response_a: str
    response_b: str
    winner: str  # model_a, model_b, or "tie"
    score_a: float
    score_b: float
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


@dataclass
class TournamentStats:
    """Tournament statistics."""

    win_matrix: np.ndarray
    model_names: List[str]
    total_matches: int
    match_history: List[MatchResult]


ArbiterCallable = Callable[[str, str, str], Tuple[str, float, float]]


class Tournament:
    """Pairwise tournament with AlphaRank and Elo-like scoring."""

    def __init__(
        self,
        model_names: List[str],
        arbiter: Optional[BaseArbiter | ArbiterCallable] = None,
        alpha: float = 1.0,
        log_dir: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ):
        self.model_names = model_names
        self.n_models = len(model_names)
        self.model_to_idx = {name: i for i, name in enumerate(model_names)}

        self._arbiter = arbiter
        self.alpha = alpha
        self.log_dir = log_dir
        self._rng = rng or random.Random()

        self.win_matrix = np.zeros((self.n_models, self.n_models), dtype=np.float64)
        self.match_count = np.zeros((self.n_models, self.n_models), dtype=np.int32)
        self.match_history: List[MatchResult] = []

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def set_arbiter(self, arbiter: BaseArbiter | ArbiterCallable) -> None:
        self._arbiter = arbiter

    def _judge(self, prompt: str, response_a: str, response_b: str) -> Tuple[str, float, float]:
        if self._arbiter is None:
            raise ValueError("Arbiter not set. Use set_arbiter() first.")

        if isinstance(self._arbiter, BaseArbiter):
            result = self._arbiter.compare(prompt, response_a, response_b)
            return result.winner, result.score_a, result.score_b

        return self._arbiter(prompt, response_a, response_b)

    def run_match(
        self,
        model_a: str,
        model_b: str,
        prompt: str,
        response_a: str,
        response_b: str,
    ) -> MatchResult:
        idx_a = self.model_to_idx[model_a]
        idx_b = self.model_to_idx[model_b]

        winner, score_a, score_b = self._judge(prompt, response_a, response_b)

        self.match_count[idx_a, idx_b] += 1
        self.match_count[idx_b, idx_a] += 1

        if winner == "a":
            self.win_matrix[idx_a, idx_b] += 1
            winner_name = model_a
        elif winner == "b":
            self.win_matrix[idx_b, idx_a] += 1
            winner_name = model_b
        else:
            self.win_matrix[idx_a, idx_b] += 0.5
            self.win_matrix[idx_b, idx_a] += 0.5
            winner_name = "tie"

        result = MatchResult(
            model_a=model_a,
            model_b=model_b,
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
            winner=winner_name,
            score_a=score_a,
            score_b=score_b,
        )

        self.match_history.append(result)
        return result

    def run_tournament(
        self,
        prompts: List[str],
        get_response: Callable[[str, str], str],
        matches_per_pair: int = 1,
        random_pairing: bool = True,
    ) -> TournamentStats:
        if random_pairing:
            for prompt in prompts:
                for _ in range(matches_per_pair):
                    model_a, model_b = self._rng.sample(self.model_names, 2)

                    response_a = get_response(model_a, prompt)
                    response_b = get_response(model_b, prompt)

                    self.run_match(model_a, model_b, prompt, response_a, response_b)
        else:
            for i, model_a in enumerate(self.model_names):
                for model_b in self.model_names[i + 1 :]:
                    for prompt in prompts[:matches_per_pair]:
                        response_a = get_response(model_a, prompt)
                        response_b = get_response(model_b, prompt)

                        self.run_match(model_a, model_b, prompt, response_a, response_b)

        return self.get_stats()

    def get_win_matrix(self) -> np.ndarray:
        return self.win_matrix.copy()

    def get_probability_matrix(self, method: str = "softmax") -> np.ndarray:
        if method == "softmax":
            prob_matrix = np.zeros_like(self.win_matrix)
            for j in range(self.n_models):
                col = self.win_matrix[:, j] * self.alpha
                exp_col = np.exp(col - np.max(col))
                prob_matrix[:, j] = exp_col / exp_col.sum()
            return prob_matrix

        if method == "normalize":
            prob_matrix = np.zeros_like(self.win_matrix)
            for i in range(self.n_models):
                for j in range(self.n_models):
                    if i != j:
                        total = self.win_matrix[i, j] + self.win_matrix[j, i]
                        if total > 0:
                            prob_matrix[i, j] = self.win_matrix[i, j] / total
                        else:
                            prob_matrix[i, j] = 0.5
            return prob_matrix

        raise ValueError(f"Unknown method: {method}")

    def get_alpharank_scores(self) -> np.ndarray:
        prob_matrix = self.get_probability_matrix(method="softmax")
        transition_matrix = prob_matrix.T.copy()

        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_matrix = transition_matrix / row_sums

        pi = np.ones(self.n_models) / self.n_models
        for _ in range(1000):
            pi_new = transition_matrix.T @ pi
            pi_new = pi_new / pi_new.sum()
            if np.allclose(pi, pi_new, atol=1e-10):
                break
            pi = pi_new

        return pi

    def get_elo_scores(self) -> np.ndarray:
        prob_matrix = self.get_probability_matrix(method="normalize")
        return prob_matrix.sum(axis=1)

    def get_rankings(self, method: str = "alpharank") -> List[Tuple[str, float]]:
        if method == "alpharank":
            scores = self.get_alpharank_scores()
        elif method == "elo":
            scores = self.get_elo_scores()
        else:
            raise ValueError(f"Unknown method: {method}")

        rankings = list(zip(self.model_names, scores))
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def get_stats(self) -> TournamentStats:
        return TournamentStats(
            win_matrix=self.win_matrix.copy(),
            model_names=self.model_names.copy(),
            total_matches=len(self.match_history),
            match_history=self.match_history.copy(),
        )

    def save_results(self, filename: str = "tournament_results.json") -> str:
        filepath = os.path.join(self.log_dir, filename) if self.log_dir else filename

        results = {
            "model_names": self.model_names,
            "win_matrix": self.win_matrix.tolist(),
            "match_count": self.match_count.tolist(),
            "alpharank_scores": self.get_alpharank_scores().tolist(),
            "elo_scores": self.get_elo_scores().tolist(),
            "rankings_alpharank": self.get_rankings("alpharank"),
            "rankings_elo": self.get_rankings("elo"),
            "total_matches": len(self.match_history),
            "match_history": [
                {
                    "model_a": m.model_a,
                    "model_b": m.model_b,
                    "prompt": m.prompt[:200] + "..." if len(m.prompt) > 200 else m.prompt,
                    "winner": m.winner,
                    "score_a": m.score_a,
                    "score_b": m.score_b,
                    "timestamp": m.timestamp,
                }
                for m in self.match_history
            ],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return filepath

    def load_results(self, filename: str = "tournament_results.json") -> None:
        filepath = os.path.join(self.log_dir, filename) if self.log_dir else filename

        with open(filepath, "r", encoding="utf-8") as f:
            results = json.load(f)

        self.model_names = results["model_names"]
        self.n_models = len(self.model_names)
        self.model_to_idx = {name: i for i, name in enumerate(self.model_names)}
        self.win_matrix = np.array(results["win_matrix"])
        self.match_count = np.array(results["match_count"])


def heuristic_arbiter(prompt: str, response_a: str, response_b: str) -> Tuple[str, float, float]:
    score_a = score_summary(response_a, prompt)
    score_b = score_summary(response_b, prompt)

    margin = 2.0
    if score_a > score_b + margin:
        winner = "a"
    elif score_b > score_a + margin:
        winner = "b"
    else:
        winner = "tie"

    return winner, score_a, score_b


def llm_arbiter_factory(
    judge_model,
    tokenizer,
    system_prompt: Optional[str] = None,
) -> Callable[[str, str, str], Tuple[str, float, float]]:
    import torch

    if system_prompt is None:
        system_prompt = """You are an expert judge evaluating the quality of text summaries.
Given a post and two summaries (A and B), determine which summary is better.
Consider: accuracy, conciseness, coherence, and how well it captures the main points.

Reply with only one character: A, B, or T (for tie).
"""

    def arbiter(prompt: str, response_a: str, response_b: str) -> Tuple[str, float, float]:
        judge_prompt = f"""{system_prompt}

POST:
{prompt}

SUMMARY A:
{response_a}

SUMMARY B:
{response_b}

Which is better? Reply with A, B, or T:"""

        inputs = tokenizer(judge_prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(judge_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = judge_model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.1,
                do_sample=False,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)) :].strip()

        first_char = response[0].upper() if response else "T"

        if first_char == "A":
            return "a", 1.0, 0.0
        if first_char == "B":
            return "b", 0.0, 1.0
        return "tie", 0.5, 0.5

    return arbiter


def remote_arbiter_factory(
    api_key: str,
    model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    base_url: str = "https://api.deepinfra.com/v1/openai",
) -> Callable[[str, str, str], Tuple[str, float, float]]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    system_prompt = """You are evaluating two summaries of a Reddit post.
Your job is to select which summary is better based on:
1. Accuracy - does it correctly summarize the main points?
2. Conciseness - is it appropriately brief?
3. Coherence - is it well-written and easy to understand?

Reply with ONLY the character 0, 1, or T (for tie).
0 means Summary 0 is better.
1 means Summary 1 is better.
T means they are equally good."""

    def arbiter(prompt: str, response_a: str, response_b: str) -> Tuple[str, float, float]:
        user_message = f"""POST:
{prompt}

SUMMARY 0:
{response_a}

SUMMARY 1:
{response_b}

Which summary is better? Reply with only 0, 1, or T:"""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=5,
                temperature=0.0,
            )

            result = response.choices[0].message.content.strip()
            first_char = result[0] if result else "T"

            if first_char == "0":
                return "a", 1.0, 0.0
            if first_char == "1":
                return "b", 0.0, 1.0
            return "tie", 0.5, 0.5

        except Exception as exc:  # pragma: no cover - network dependent
            print(f"API error: {exc}")
            return "tie", 0.5, 0.5

    return arbiter
