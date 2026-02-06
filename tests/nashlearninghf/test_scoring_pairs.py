from nashlearninghf.scoring import create_better_preference_pairs


def test_create_better_preference_pairs():
    prompts = ["PROMPT 1", "PROMPT 2"]
    generated = {
        "p1": ["Same summary.", "Same summary."],
        "p2": ["Same summary.", "Same summary."],
    }

    pairs = create_better_preference_pairs(prompts, generated, pairs_per_prompt=1)
    assert pairs == []


def test_create_better_preference_pairs_adds_pairs():
    prompts = [
        "POST: alpha beta gamma TL;DR:",
        "POST: alpha beta gamma TL;DR:",
    ]
    generated = {
        "p1": ["alpha beta gamma.", "alpha beta gamma."],
        "p2": ["bad", "bad"],
    }

    pairs = create_better_preference_pairs(prompts, generated, pairs_per_prompt=1)
    assert pairs
