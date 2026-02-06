"""Run a local mock tournament (exp1)."""

from __future__ import annotations

from nashlearninghf import HeuristicArbiter, InMemoryPromptProvider, TournamentExperiment
from nashlearninghf.models.mock import create_mock_models


def main() -> None:
    prompts = [
        "SUBREDDIT: r/relationships\nTITLE: Moving in together\nPOST: We are considering moving in together after a year...\nTL;DR:",
        "SUBREDDIT: r/technology\nTITLE: New phone choice\nPOST: I am choosing between two phones with different cameras...\nTL;DR:",
        "SUBREDDIT: r/finance\nTITLE: Budget reset\nPOST: I need to rebuild my budget after a big move...\nTL;DR:",
    ]

    models = create_mock_models(
        names=["mock_a", "mock_b", "mock_c"],
        styles=["verbose", "concise", "balanced"],
        strengths=[0.6, 0.7, 0.8],
    )

    arbiter = HeuristicArbiter(margin=2.0)
    provider = InMemoryPromptProvider(prompts=prompts)

    experiment = TournamentExperiment(
        models=models,
        arbiter=arbiter,
        prompt_provider=provider,
        matches_per_pair=2,
        random_pairing=True,
        alpha=1.0,
        seed=42,
    )

    stats = experiment.run()

    print("Total matches:", stats.total_matches)

    rankings = experiment_run_rankings(stats)
    print("Rankings:")
    for name, score in rankings:
        print(f"  {name}: {score:.4f}")


def experiment_run_rankings(stats):
    from nashlearninghf.tournament import Tournament

    tournament = Tournament(stats.model_names)
    tournament.win_matrix = stats.win_matrix
    return tournament.get_rankings("alpharank")


if __name__ == "__main__":
    main()
