"""
NLHF Tournament Experiment v3
Package for running model tournaments and AlphaRank evaluation.

Modules:
- tournament.py: Tournament class with AlphaRank/Elo scoring
- scoring.py: Heuristic + LLM scoring functions (ppotune compatible)
- visualization.py: Plotting utilities
- experiment.py: Core experiment logic
- run_experiment.py: CLI runner

Arbiters:
- heuristic: Fast local scoring based on text features
- llm: Remote LLM arbiter via OpenAI-compatible API
- hybrid: Combination of heuristic + LLM

Usage:
    # Run from command line:
    python run_experiment.py --prompts 50 --matches 2 --arbiter heuristic
    python run_experiment.py --prompts 50 --arbiter llm --llm-model meta-llama/Llama-3.3-70B-Instruct-Turbo
    
    # Or import and use:
    from experiment import ExperimentConfig, run_experiment
    config = ExperimentConfig(num_prompts=50, arbiter_type="heuristic")
    results = run_experiment(config)
"""

from .tournament import Tournament, TournamentStats, MatchResult
from .tournament import heuristic_arbiter, llm_arbiter_factory, remote_arbiter_factory
from .scoring import (
    score_summary, 
    compare_responses,
    create_heuristic_arbiter,
    create_llm_arbiter,
    create_hybrid_arbiter,
    RemoteTLDRArbiter,
    batch_judge
)
from .visualization import (
    plot_win_matrix,
    plot_probability_matrix,
    plot_rankings,
    plot_score_distribution,
    plot_match_history,
    plot_pairwise_comparison,
    create_summary_dashboard
)
from .experiment import (
    ExperimentConfig,
    ExperimentResults,
    run_experiment,
    print_results_summary,
    analyze_upsets,
    get_upset_statistics,
    create_arbiter
)

__all__ = [
    # Tournament
    "Tournament",
    "TournamentStats", 
    "MatchResult",
    "heuristic_arbiter",
    "llm_arbiter_factory",
    "remote_arbiter_factory",
    # Scoring & Arbiters
    "score_summary",
    "compare_responses",
    "create_heuristic_arbiter",
    "create_llm_arbiter",
    "create_hybrid_arbiter",
    "RemoteTLDRArbiter",
    "batch_judge",
    # Visualization
    "plot_win_matrix",
    "plot_probability_matrix",
    "plot_rankings",
    "plot_score_distribution",
    "plot_match_history",
    "plot_pairwise_comparison",
    "create_summary_dashboard",
    # Experiment
    "ExperimentConfig",
    "ExperimentResults",
    "run_experiment",
    "create_arbiter",
    "print_results_summary",
    "analyze_upsets",
    "get_upset_statistics",
]
