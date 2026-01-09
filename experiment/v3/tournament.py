"""
Tournament System for comparing multiple LLM models.
Implements pairwise comparison, win matrix calculation, and AlphaRank scoring.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
import random
import json
import os
from datetime import datetime


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
    """Statistics for a tournament."""
    win_matrix: np.ndarray  # [i, j] = number of wins for model i against model j
    model_names: List[str]
    total_matches: int
    match_history: List[MatchResult]


class Tournament:
    """
    Tournament system for comparing multiple models.
    
    Supports:
    - Pairwise comparison with configurable arbiter
    - Win matrix calculation
    - Probability matrix via softmax
    - AlphaRank scoring (eigenvector method)
    - Elo-like scoring (row sum method)
    """
    
    def __init__(
        self,
        model_names: List[str],
        arbiter: Optional[Callable[[str, str, str], Tuple[str, float, float]]] = None,
        alpha: float = 1.0,  # Softmax temperature for AlphaRank
        log_dir: Optional[str] = None
    ):
        """
        Args:
            model_names: List of model identifiers
            arbiter: Function(prompt, response_a, response_b) -> (winner, score_a, score_b)
                     winner is "a", "b", or "tie"
            alpha: Temperature parameter for softmax in AlphaRank
            log_dir: Directory to save match logs
        """
        self.model_names = model_names
        self.n_models = len(model_names)
        self.model_to_idx = {name: i for i, name in enumerate(model_names)}
        
        self.arbiter = arbiter
        self.alpha = alpha
        self.log_dir = log_dir
        
        # Win matrix: [i, j] = number of times model i beat model j
        self.win_matrix = np.zeros((self.n_models, self.n_models), dtype=np.float64)
        self.match_count = np.zeros((self.n_models, self.n_models), dtype=np.int32)
        
        self.match_history: List[MatchResult] = []
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    def set_arbiter(self, arbiter: Callable[[str, str, str], Tuple[str, float, float]]):
        """Set the arbiter function for judging matches."""
        self.arbiter = arbiter
    
    def run_match(
        self,
        model_a: str,
        model_b: str,
        prompt: str,
        response_a: str,
        response_b: str
    ) -> MatchResult:
        """
        Run a single match between two models.
        
        Args:
            model_a: First model name
            model_b: Second model name
            prompt: The prompt given to both models
            response_a: Response from model_a
            response_b: Response from model_b
        
        Returns:
            MatchResult with winner and scores
        """
        if self.arbiter is None:
            raise ValueError("Arbiter not set. Use set_arbiter() first.")
        
        idx_a = self.model_to_idx[model_a]
        idx_b = self.model_to_idx[model_b]
        
        winner, score_a, score_b = self.arbiter(prompt, response_a, response_b)
        
        # Update win matrix
        self.match_count[idx_a, idx_b] += 1
        self.match_count[idx_b, idx_a] += 1
        
        if winner == "a":
            self.win_matrix[idx_a, idx_b] += 1
            winner_name = model_a
        elif winner == "b":
            self.win_matrix[idx_b, idx_a] += 1
            winner_name = model_b
        else:  # tie
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
            score_b=score_b
        )
        
        self.match_history.append(result)
        
        return result
    
    def run_tournament(
        self,
        prompts: List[str],
        get_response: Callable[[str, str], str],
        matches_per_pair: int = 1,
        random_pairing: bool = True
    ) -> TournamentStats:
        """
        Run a full tournament with all model pairs.
        
        Args:
            prompts: List of prompts to use
            get_response: Function(model_name, prompt) -> response
            matches_per_pair: Number of matches for each model pair
            random_pairing: If True, randomly select pairs; otherwise round-robin
        
        Returns:
            TournamentStats with results
        """
        if random_pairing:
            # Random selection of pairs for each prompt
            for prompt in prompts:
                for _ in range(matches_per_pair):
                    # Select two different models randomly
                    model_a, model_b = random.sample(self.model_names, 2)
                    
                    response_a = get_response(model_a, prompt)
                    response_b = get_response(model_b, prompt)
                    
                    self.run_match(model_a, model_b, prompt, response_a, response_b)
        else:
            # Round-robin: all pairs
            for i, model_a in enumerate(self.model_names):
                for model_b in self.model_names[i+1:]:
                    for prompt in prompts[:matches_per_pair]:
                        response_a = get_response(model_a, prompt)
                        response_b = get_response(model_b, prompt)
                        
                        self.run_match(model_a, model_b, prompt, response_a, response_b)
        
        return self.get_stats()
    
    def get_win_matrix(self) -> np.ndarray:
        """Get the raw win count matrix."""
        return self.win_matrix.copy()
    
    def get_probability_matrix(self, method: str = "softmax") -> np.ndarray:
        """
        Convert win matrix to probability matrix.
        
        Args:
            method: "softmax" (column-wise) or "normalize" (simple ratio)
        
        Returns:
            Probability matrix where [i, j] = P(i beats j)
        """
        if method == "softmax":
            # Apply softmax column-wise (for each opponent j)
            prob_matrix = np.zeros_like(self.win_matrix)
            for j in range(self.n_models):
                col = self.win_matrix[:, j] * self.alpha
                # Softmax
                exp_col = np.exp(col - np.max(col))  # subtract max for numerical stability
                prob_matrix[:, j] = exp_col / exp_col.sum()
            return prob_matrix
        
        elif method == "normalize":
            # Simple normalization: wins / (wins + losses)
            prob_matrix = np.zeros_like(self.win_matrix)
            for i in range(self.n_models):
                for j in range(self.n_models):
                    if i != j:
                        total = self.win_matrix[i, j] + self.win_matrix[j, i]
                        if total > 0:
                            prob_matrix[i, j] = self.win_matrix[i, j] / total
                        else:
                            prob_matrix[i, j] = 0.5  # No data, assume equal
            return prob_matrix
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_alpharank_scores(self) -> np.ndarray:
        """
        Calculate AlphaRank scores using eigenvector method.
        
        The stationary distribution of a Markov chain where transition
        probabilities are based on win probabilities.
        
        Returns:
            Array of scores for each model (sums to 1)
        """
        prob_matrix = self.get_probability_matrix(method="softmax")
        
        # Create transition matrix for Markov chain
        # Each row represents transition probabilities from state i
        # We use the transpose of prob_matrix and normalize rows
        transition_matrix = prob_matrix.T.copy()
        
        # Normalize rows to sum to 1
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_matrix / row_sums
        
        # Find stationary distribution (left eigenvector with eigenvalue 1)
        # Power iteration method
        pi = np.ones(self.n_models) / self.n_models
        for _ in range(1000):
            pi_new = transition_matrix.T @ pi
            pi_new = pi_new / pi_new.sum()
            if np.allclose(pi, pi_new, atol=1e-10):
                break
            pi = pi_new
        
        return pi
    
    def get_elo_scores(self) -> np.ndarray:
        """
        Calculate Elo-like scores (average win probability).
        
        Simple method: sum each row of probability matrix.
        Higher means better average performance against all opponents.
        
        Returns:
            Array of scores for each model
        """
        prob_matrix = self.get_probability_matrix(method="normalize")
        # Sum rows (excluding diagonal)
        scores = prob_matrix.sum(axis=1)
        return scores
    
    def get_rankings(self, method: str = "alpharank") -> List[Tuple[str, float]]:
        """
        Get ranked list of models.
        
        Args:
            method: "alpharank" or "elo"
        
        Returns:
            List of (model_name, score) tuples, sorted by score descending
        """
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
        """Get tournament statistics."""
        return TournamentStats(
            win_matrix=self.win_matrix.copy(),
            model_names=self.model_names.copy(),
            total_matches=len(self.match_history),
            match_history=self.match_history.copy()
        )
    
    def save_results(self, filename: str = "tournament_results.json"):
        """Save tournament results to file."""
        if self.log_dir:
            filepath = os.path.join(self.log_dir, filename)
        else:
            filepath = filename
        
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
                    "timestamp": m.timestamp
                }
                for m in self.match_history
            ]
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def load_results(self, filename: str = "tournament_results.json"):
        """Load tournament results from file."""
        if self.log_dir:
            filepath = os.path.join(self.log_dir, filename)
        else:
            filepath = filename
        
        with open(filepath, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        self.model_names = results["model_names"]
        self.n_models = len(self.model_names)
        self.model_to_idx = {name: i for i, name in enumerate(self.model_names)}
        self.win_matrix = np.array(results["win_matrix"])
        self.match_count = np.array(results["match_count"])


# ==================== Arbiters ====================

def heuristic_arbiter(prompt: str, response_a: str, response_b: str) -> Tuple[str, float, float]:
    """
    Simple heuristic arbiter based on improved_preference_scoring.
    
    Returns:
        Tuple of (winner, score_a, score_b)
        winner is "a", "b", or "tie"
    """
    from v3.scoring import score_summary
    
    score_a = score_summary(response_a, prompt)
    score_b = score_summary(response_b, prompt)
    
    # Determine winner with small margin for ties
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
    system_prompt: Optional[str] = None
) -> Callable[[str, str, str], Tuple[str, float, float]]:
    """
    Create an LLM-based arbiter.
    
    Args:
        judge_model: HuggingFace model for judging
        tokenizer: Tokenizer for the model
        system_prompt: Optional system prompt for judging
    
    Returns:
        Arbiter function
    """
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
                do_sample=False
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the generated part
        response = response[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()
        
        first_char = response[0].upper() if response else "T"
        
        if first_char == "A":
            return "a", 1.0, 0.0
        elif first_char == "B":
            return "b", 0.0, 1.0
        else:
            return "tie", 0.5, 0.5
    
    return arbiter


def remote_arbiter_factory(
    api_key: str,
    model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    base_url: str = "https://api.deepinfra.com/v1/openai"
) -> Callable[[str, str, str], Tuple[str, float, float]]:
    """
    Create a remote LLM arbiter using OpenAI-compatible API.
    
    Args:
        api_key: API key for the service
        model: Model name to use
        base_url: API base URL
    
    Returns:
        Arbiter function
    """
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
                    {"role": "user", "content": user_message}
                ],
                max_tokens=5,
                temperature=0.0
            )
            
            result = response.choices[0].message.content.strip()
            first_char = result[0] if result else "T"
            
            if first_char == "0":
                return "a", 1.0, 0.0
            elif first_char == "1":
                return "b", 0.0, 1.0
            else:
                return "tie", 0.5, 0.5
                
        except Exception as e:
            print(f"API error: {e}")
            return "tie", 0.5, 0.5
    
    return arbiter
