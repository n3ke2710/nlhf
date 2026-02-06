from nashlearninghf.scoring import score_summary


def test_score_summary_prefers_reasonable_summary():
    prompt = (
        "SUBREDDIT: r/test\nTITLE: Example\nPOST: This is a long post about a situation "
        "with multiple details and outcomes that need summarization.\nTL;DR:"
    )

    good = "The post describes a situation with details and a clear outcome."
    bad = "help me"

    assert score_summary(good, prompt) > score_summary(bad, prompt)


def test_score_summary_branches():
    prompt = "POST: " + ("word " * 200) + "TL;DR:"

    ideal = "word " * 20
    score_summary(ideal, prompt)

    comma_end = "This ends with comma,"
    score_summary(comma_end, prompt)

    with_newlines = "Line one.\n\nLine two."
    score_summary(with_newlines, prompt)

    with_ellipsis = "This has... too many dots..."
    score_summary(with_ellipsis, prompt)

    with_verb = "I need help with the situation."
    score_summary(with_verb, prompt)
