from nashlearninghf.arbiters.heuristic import HeuristicArbiter


def test_heuristic_arbiter_basic():
    arbiter = HeuristicArbiter(margin=0.1)
    prompt = "This is a long prompt with many details about a situation."

    response_a = "Summary with details and clear ending."
    response_b = "Short."

    result = arbiter.compare(prompt, response_a, response_b)
    assert result.winner in {"a", "b", "tie"}
    assert result.score_a >= 0
    assert result.score_b >= 0


def test_heuristic_tie_margin():
    arbiter = HeuristicArbiter(margin=1000.0)
    prompt = "Prompt text with some content."
    response_a = "Summary A."
    response_b = "Summary B."

    result = arbiter.compare(prompt, response_a, response_b)
    assert result.winner == "tie"


def test_heuristic_handles_empty_or_error_summary():
    arbiter = HeuristicArbiter(margin=0.1)
    prompt = "Some prompt text with enough words to score."

    result_error = arbiter.compare(prompt, "[Error generating response]", "Ok summary.")
    assert result_error.score_a == 0.0

    result_empty = arbiter.compare(prompt, "", "Ok summary.")
    assert result_empty.score_a == 0.0


def test_heuristic_internal_branches():
    arbiter = HeuristicArbiter(margin=0.1)

    long_original = "word " * 200
    tiny_summary = "short"
    arbiter._score_summary(long_original, tiny_summary)

    stopword_original = "the and or but"
    summary = "Cats are great."
    arbiter._score_summary(stopword_original, summary)

    whitespace_summary = "   "
    assert arbiter._score_summary("Some text", whitespace_summary) == 0.0

    assert arbiter._readability_score("") == 0.0

    long_sentence = "word " * 40
    assert arbiter._readability_score(long_sentence.strip()) >= 0
