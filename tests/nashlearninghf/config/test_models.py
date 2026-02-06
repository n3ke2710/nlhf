import os

from nashlearninghf.config.models import load_models_from_yaml


def test_load_models_from_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    yaml_text = """
defaults:
  api_key: ${OPENAI_API_KEY}
  base_url: https://api.openai.com/v1
  max_tokens: 55
  temperature: 0.3

models:
  - name: mock-0
    type: mock
    strength: 0.6
  - name: mock-1
    type: mock
    strength: 0.7
    style: concise
"""

    path = tmp_path / "models.yaml"
    path.write_text(yaml_text)

    config = load_models_from_yaml(str(path))
    assert len(config.models) == 2
    assert config.models[0].api_key == "test-key"
    assert config.models[1].style == "concise"
    assert config.get_model_names() == ["mock-0", "mock-1"]

    models = config.create_models()
    assert set(models.keys()) == {"mock-0", "mock-1"}
