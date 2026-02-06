import numpy as np

from nashlearninghf.tournament import Tournament, heuristic_arbiter


def test_run_match_and_matrices():
    tournament = Tournament(["a", "b"], arbiter=lambda p, ra, rb: ("a", 1.0, 0.0))
    result = tournament.run_match("a", "b", "prompt", "ra", "rb")

    assert result.winner == "a"
    assert tournament.win_matrix[0, 1] == 1
    assert tournament.win_matrix[1, 0] == 0
    assert tournament.match_count[0, 1] == 1
    assert tournament.match_count[1, 0] == 1


def test_run_match_winner_b_path():
    tournament = Tournament(["a", "b"], arbiter=lambda p, ra, rb: ("b", 0.0, 1.0))
    result = tournament.run_match("a", "b", "prompt", "ra", "rb")
    assert result.winner == "b"
    assert tournament.win_matrix[1, 0] == 1


def test_set_arbiter_and_judge_errors():
    from nashlearninghf.arbiters.base import ArbiterResult, BaseArbiter

    class DummyArbiter(BaseArbiter):
        def compare(self, prompt, response_a, response_b):
            return ArbiterResult(winner="a", score_a=1.0, score_b=0.0)

    tournament = Tournament(["a", "b"])
    try:
        tournament.run_match("a", "b", "p", "ra", "rb")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    tournament.set_arbiter(DummyArbiter())
    result = tournament.run_match("a", "b", "p", "ra", "rb")
    assert result.winner == "a"


def test_run_tournament_random_pairing():
    tournament = Tournament(["a", "b", "c"], arbiter=lambda p, ra, rb: ("tie", 0.5, 0.5))

    def responder(name, prompt):
        return f"{name}:{prompt}"

    stats = tournament.run_tournament(
        prompts=["p1", "p2"],
        get_response=responder,
        matches_per_pair=1,
        random_pairing=True,
    )
    assert stats.total_matches == 2


def test_probability_matrices_and_rankings():
    tournament = Tournament(["a", "b", "c"], arbiter=lambda p, ra, rb: ("tie", 0.5, 0.5))
    tournament.win_matrix = np.array(
        [
            [0, 3, 2],
            [1, 0, 4],
            [2, 0, 0],
        ],
        dtype=float,
    )

    softmax = tournament.get_probability_matrix(method="softmax")
    normalize = tournament.get_probability_matrix(method="normalize")

    assert softmax.shape == (3, 3)
    assert normalize.shape == (3, 3)

    rankings_alpha = tournament.get_rankings(method="alpharank")
    rankings_elo = tournament.get_rankings(method="elo")

    assert len(rankings_alpha) == 3
    assert len(rankings_elo) == 3

    win_matrix = tournament.get_win_matrix()
    assert win_matrix.shape == (3, 3)

    empty_tournament = Tournament(["a", "b"])
    normalize = empty_tournament.get_probability_matrix(method="normalize")
    assert normalize[0, 1] == 0.5

    try:
        tournament.get_probability_matrix(method="unknown")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        tournament.get_rankings(method="unknown")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_save_and_load_results(tmp_path):
    tournament = Tournament(["a", "b"], arbiter=lambda p, ra, rb: ("a", 1.0, 0.0), log_dir=str(tmp_path))
    tournament.run_match("a", "b", "prompt", "ra", "rb")

    path = tournament.save_results("results.json")

    new_tournament = Tournament(["x", "y"], log_dir=str(tmp_path))
    new_tournament.load_results("results.json")

    assert new_tournament.model_names == ["a", "b"]
    assert new_tournament.win_matrix.shape == (2, 2)


def test_heuristic_arbiter_function():
    winner, score_a, score_b = heuristic_arbiter("Prompt text", "Good summary.", "Bad")
    assert winner in {"a", "b", "tie"}
    assert score_a != score_b or winner == "tie"

    winner, score_a, score_b = heuristic_arbiter("Prompt text", "Same", "Same")
    assert winner == "tie"

    prompt = "SUBREDDIT: r/test\nTITLE: T\nPOST: " + ("alpha " * 100) + "\nTL;DR:"
    winner, score_a, score_b = heuristic_arbiter(prompt, "Alpha alpha alpha.", "")
    assert winner == "a"
