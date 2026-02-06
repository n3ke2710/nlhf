from nashlearninghf import HeuristicArbiter, InMemoryPromptProvider, TournamentExperiment
from nashlearninghf.models.mock import create_mock_models


def test_tournament_experiment_runs():
    prompts = [
        "SUBREDDIT: r/test\nTITLE: A\nPOST: Some post details here.\nTL;DR:",
        "SUBREDDIT: r/test\nTITLE: B\nPOST: Another post with info.\nTL;DR:",
    ]

    models = create_mock_models(
        names=["m1", "m2"],
        styles=["balanced", "concise"],
        strengths=[0.6, 0.7],
    )

    experiment = TournamentExperiment(
        models=models,
        arbiter=HeuristicArbiter(margin=1.0),
        prompt_provider=InMemoryPromptProvider(prompts=prompts),
        matches_per_pair=2,
        random_pairing=False,
        seed=123,
    )

    stats = experiment.run()
    assert stats.total_matches == 2
    assert len(stats.match_history) == 2
