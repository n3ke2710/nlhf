from nashlearninghf.arbiters.base import ArbiterResult, BaseArbiter


class DummyArbiter(BaseArbiter):
    def compare(self, prompt: str, response_a: str, response_b: str) -> ArbiterResult:
        return ArbiterResult(winner="a", score_a=1.0, score_b=0.0)


def test_base_arbiter_call():
    arbiter = DummyArbiter()
    result = arbiter("p", "a", "b")
    assert result.winner == "a"
