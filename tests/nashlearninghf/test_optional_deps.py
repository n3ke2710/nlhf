import sys
from types import SimpleNamespace

import pytest

from nashlearninghf.models.base import ModelConfig
from nashlearninghf.models.api import APIModel
from nashlearninghf.arbiters.llm import LLMArbiter
from nashlearninghf.models.local import LocalModel, LoRAModel, load_local_model, load_lora_model


class FakeOpenAI:
    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.models = SimpleNamespace(list=lambda: [])

    def _create(self, *args, **kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])


def test_api_model_generate_and_available(monkeypatch):
    fake_module = SimpleNamespace(OpenAI=FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    model = APIModel(ModelConfig(name="api", model_type="api", model_id="x"))
    assert model.is_available() is True
    assert model.generate("prompt").lower() == "ok"


def test_llm_arbiter_compare_branches(monkeypatch):
    class FakeOpenAIA:
        def __init__(self, *args, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, *args, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="A"))])

    class FakeOpenAIB(FakeOpenAIA):
        def _create(self, *args, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="B"))])

    class FakeOpenAITie(FakeOpenAIA):
        def _create(self, *args, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="TIE"))])

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAIA))
    arbiter = LLMArbiter(api_key="k")
    assert arbiter.compare("prompt", "a", "b").winner == "a"

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAIB))
    arbiter = LLMArbiter(api_key="k")
    assert arbiter.compare("prompt", "a", "b").winner == "b"

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAITie))
    arbiter = LLMArbiter(api_key="k")
    assert arbiter.compare("prompt", "a", "b").winner == "tie"


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def test_local_and_lora_model_generate(monkeypatch):
    class FakeTokenizer:
        eos_token = "<eos>"
        pad_token = "<eos>"
        pad_token_id = 0

        def __call__(self, *args, **kwargs):
            class InputDict(dict):
                def to(self, device):
                    return self

            return InputDict(input_ids=[1, 2, 3])

        def decode(self, output, skip_special_tokens=True):
            return "TL;DR: ok"

    class FakeModel:
        def __init__(self):
            self._device = "cpu"

        def parameters(self):
            return iter([SimpleNamespace(device=self._device)])

        def generate(self, **kwargs):
            return ["output"]

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path):
            return FakeTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeModel()

    class FakeBitsAndBytesConfig:
        def __init__(self, *args, **kwargs):
            pass

    class FakePeftModel:
        @staticmethod
        def from_pretrained(base_model, adapter_path):
            return base_model

    fake_torch = SimpleNamespace(bfloat16="bf16", no_grad=lambda: FakeNoGrad())
    fake_transformers = SimpleNamespace(
        AutoTokenizer=FakeAutoTokenizer,
        AutoModelForCausalLM=FakeAutoModel,
        BitsAndBytesConfig=FakeBitsAndBytesConfig,
    )
    fake_peft = SimpleNamespace(PeftModel=FakePeftModel)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    local = LocalModel(
        ModelConfig(
            name="local",
            model_type="local",
            model_path="/tmp",
            quantization="4bit",
        )
    )
    assert local.generate("prompt") == "ok"
    assert local.is_available() is True

    lora = LoRAModel(
        ModelConfig(
            name="lora",
            model_type="lora",
            base_model_path="/tmp/base",
            model_path="/tmp/adapter",
            quantization="8bit",
        )
    )
    assert lora.generate("prompt") == "ok"
    assert lora.is_available() is True

    local_loaded = load_local_model("l2", "/tmp", quantization="8bit")
    assert local_loaded.is_available() is True

    lora_loaded = load_lora_model("l3", "/tmp/base", "/tmp/adapter", quantization="4bit")
    assert lora_loaded.is_available() is True
