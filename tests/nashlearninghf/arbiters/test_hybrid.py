import sys

from nashlearninghf.arbiters.base import ArbiterResult
from nashlearninghf.arbiters.heuristic import HeuristicArbiter
from nashlearninghf.arbiters.hybrid import HybridArbiter


class StubLLM:
    def __init__(self):
        self.calls = 0

    def compare(self, prompt: str, response_a: str, response_b: str) -> ArbiterResult:
        self.calls += 1
        return ArbiterResult(winner="b", score_a=0.0, score_b=1.0, confidence=1.0)


def test_hybrid_uses_heuristic_when_confident():
    heuristic = HeuristicArbiter(margin=0.1)
    stub_llm = StubLLM()
    arbiter = HybridArbiter(
        heuristic=heuristic,
        llm=stub_llm,
        llm_threshold=0.0,
        heuristic_weight=0.5,
    )

    prompt = "Long prompt text for scoring."
    response_a = "Good summary with detail."
    response_b = "Bad."

    result = arbiter.compare(prompt, response_a, response_b)
    assert result.winner in {"a", "b", "tie"}
    assert stub_llm.calls == 0


def test_hybrid_skips_llm_when_threshold_high():
    heuristic = HeuristicArbiter(margin=0.1)
    stub_llm = StubLLM()
    arbiter = HybridArbiter(
        heuristic=heuristic,
        llm=stub_llm,
        llm_threshold=1e9,
        heuristic_weight=0.5,
    )

    prompt = "Prompt text with details."
    response_a = "Summary A."
    response_b = "Summary B."

    result = arbiter.compare(prompt, response_a, response_b)
    assert result.winner in {"a", "b", "tie"}
    assert stub_llm.calls == 1


def test_hybrid_handles_zero_scores_and_tie():
    class ZeroHeuristic(HeuristicArbiter):
        def compare(self, prompt, response_a, response_b):
            return ArbiterResult(winner="tie", score_a=0.0, score_b=0.0, confidence=0.0)

    class TieLLM:
        def compare(self, prompt, response_a, response_b):
            return ArbiterResult(winner="tie", score_a=0.5, score_b=0.5, confidence=0.0)

    arbiter = HybridArbiter(
        heuristic=ZeroHeuristic(),
        llm=TieLLM(),
        llm_threshold=1.0,
        heuristic_weight=0.5,
    )
    result = arbiter.compare("p", "a", "b")
    assert result.winner == "tie"


def test_create_arbiter_invalid():
    from nashlearninghf.arbiters.hybrid import create_arbiter

    try:
        create_arbiter("unknown")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_create_arbiter_heuristic():
    from nashlearninghf.arbiters.hybrid import create_arbiter

    arbiter = create_arbiter("heuristic")
    assert isinstance(arbiter, HeuristicArbiter)


def test_hybrid_prefers_a_when_scores_high():
    class SimpleHeuristic(HeuristicArbiter):
        def compare(self, prompt, response_a, response_b):
            return ArbiterResult(winner="tie", score_a=10.0, score_b=5.0, confidence=1.0)

    class PreferA:
        def compare(self, prompt, response_a, response_b):
            return ArbiterResult(winner="a", score_a=1.0, score_b=0.0, confidence=1.0)

    arbiter = HybridArbiter(
        heuristic=SimpleHeuristic(),
        llm=PreferA(),
        llm_threshold=100.0,
        heuristic_weight=0.5,
    )
    result = arbiter.compare("p", "a", "b")
    assert result.winner == "a"


def test_create_arbiter_llm_and_hybrid(monkeypatch):
    from types import SimpleNamespace
    from nashlearninghf.arbiters.hybrid import create_arbiter

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, *args, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="A"))])

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    llm_arbiter = create_arbiter("llm", api_key="k")
    assert llm_arbiter is not None

    hybrid = create_arbiter("hybrid", api_key="k")
    assert hybrid is not None
