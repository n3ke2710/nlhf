import sys
from types import SimpleNamespace

from nashlearninghf.tournament import llm_arbiter_factory, remote_arbiter_factory


def test_llm_arbiter_factory(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.device = "cpu"

        def generate(self, **kwargs):
            return ["A"]

    class FakeTokenizer:
        def __call__(self, text, return_tensors, truncation, max_length):
            class InputTensor:
                def to(self, device):
                    return self

                def __getitem__(self, idx):
                    return "token"

            return {"input_ids": InputTensor()}

        def decode(self, output, skip_special_tokens=True):
            if isinstance(output, list):
                return ""
            if output == "token":
                return ""
            return "A"

    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_torch = SimpleNamespace(no_grad=lambda: FakeNoGrad())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    arbiter = llm_arbiter_factory(FakeModel(), FakeTokenizer())
    winner, score_a, score_b = arbiter("prompt", "a", "b")
    assert winner in {"a", "b", "tie"}


def test_remote_arbiter_factory(monkeypatch):
    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, *args, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="0"))])

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    arbiter = remote_arbiter_factory(api_key="k")
    winner, score_a, score_b = arbiter("prompt", "a", "b")
    assert winner == "a"
    assert score_a == 1.0


def test_llm_arbiter_factory_b_branch(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.device = "cpu"

        def generate(self, **kwargs):
            return ["B"]

    class FakeTokenizer:
        def __call__(self, text, return_tensors, truncation, max_length):
            class InputTensor:
                def to(self, device):
                    return self

                def __getitem__(self, idx):
                    return "token"

            return {"input_ids": InputTensor()}

        def decode(self, output, skip_special_tokens=True):
            if isinstance(output, list):
                return ""
            if output == "token":
                return ""
            return "B"

    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_torch = SimpleNamespace(no_grad=lambda: FakeNoGrad())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    arbiter = llm_arbiter_factory(FakeModel(), FakeTokenizer())
    winner, score_a, score_b = arbiter("prompt", "a", "b")
    assert winner == "b"
    assert score_b == 1.0


def test_remote_arbiter_factory_b_and_tie(monkeypatch):
    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, *args, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="1"))])

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    arbiter = remote_arbiter_factory(api_key="k")
    winner, score_a, score_b = arbiter("prompt", "a", "b")
    assert winner == "b"
    assert score_b == 1.0

    class FakeOpenAITie(FakeOpenAI):
        def _create(self, *args, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="T"))])

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAITie))
    arbiter = remote_arbiter_factory(api_key="k")
    winner, score_a, score_b = arbiter("prompt", "a", "b")
    assert winner == "tie"


def test_llm_arbiter_factory_tie_branch(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.device = "cpu"

        def generate(self, **kwargs):
            return [""]

    class FakeTokenizer:
        def __call__(self, text, return_tensors, truncation, max_length):
            class InputTensor:
                def to(self, device):
                    return self

                def __getitem__(self, idx):
                    return ""

            return {"input_ids": InputTensor()}

        def decode(self, output, skip_special_tokens=True):
            return ""

    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_torch = SimpleNamespace(no_grad=lambda: FakeNoGrad())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    arbiter = llm_arbiter_factory(FakeModel(), FakeTokenizer())
    winner, score_a, score_b = arbiter("prompt", "a", "b")
    assert winner == "tie"
