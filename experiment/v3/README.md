# NLHF Tournament v3

Tournament-based comparison of LLM models using AlphaRank and Elo scoring.

## Structure

```
v3/
├── main.py              # Main CLI entry point
├── run.sh               # Quick run script
├── .env                 # API keys (gitignored)
├── requirements.txt     # Dependencies
│
├── models/              # Model implementations
│   ├── __init__.py
│   ├── base.py          # BaseModel, ModelRegistry
│   ├── api.py           # OpenAI, Anthropic, DeepInfra
│   ├── local.py         # Local HuggingFace, LoRA
│   └── mock.py          # Mock models for testing
│
├── arbiters/            # Comparison functions
│   ├── __init__.py
│   ├── base.py          # BaseArbiter interface
│   ├── heuristic.py     # Fast local heuristics
│   ├── llm.py           # LLM-based comparison
│   └── hybrid.py        # Combined approach
│
├── config/              # Configuration
│   ├── __init__.py
│   ├── experiment.py    # ExperimentConfig
│   ├── models.py        # ModelsConfig, YAML loader
│   └── models.yaml      # Model definitions
│
├── tournament.py        # Tournament logic, AlphaRank, Elo
├── visualization.py     # Plotting utilities
│
└── output/              # Results (gitignored)
    ├── logs/
    └── graphics/
```

## Quick Start

```bash
# 1. Setup
cp .env.example .env
# Edit .env and add OPENAI_API_KEY

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
./run.sh mock      # Test with mock models
./run.sh api       # Compare OpenAI models (heuristic)
./run.sh api-llm   # Compare with LLM arbiter
./run.sh full      # Full experiment
```

## Usage

### Command Line

```bash
# Compare specific models
python main.py --models gpt-4o-mini gpt-3.5-turbo --arbiter heuristic

# Use config file
python main.py --config config/models.yaml --arbiter llm

# Full options
python main.py \
    --models gpt-4o gpt-4o-mini gpt-3.5-turbo \
    --prompts 50 \
    --matches 3 \
    --arbiter llm \
    --arbiter-model gpt-4o \
    --output ./output/experiment1
```

### Config File

Edit `config/models.yaml` to define models:

```yaml
defaults:
  api_key: ${OPENAI_API_KEY}
  base_url: https://api.openai.com/v1

models:
  - name: gpt-4o-mini
    type: openai
    model_id: gpt-4o-mini
  
  - name: llama-70b
    type: deepinfra
    model_id: meta-llama/Llama-3.3-70B-Instruct-Turbo
    api_key: ${DEEPINFRA_API_KEY}
  
  - name: my-local-model
    type: local
    model_path: ~/.tune/models/Llama-3.2-3B/
    quantization: 4bit
```

## Adding New Models

### API Models

```python
from models.base import ModelConfig
from models.api import OpenAIModel

model = OpenAIModel(ModelConfig(
    name="my-model",
    model_type="openai",
    model_id="gpt-4o",
    api_key="sk-..."
))
```

### Local Models

```python
from models.local import load_lora_model

model = load_lora_model(
    name="my-finetuned",
    base_model_path="~/.tune/models/Llama-3.2-3B-Base/",
    adapter_path="~/.tune/checkpoints/my-lora/",
    quantization="4bit"
)
```

### Custom Model Type

```python
from models.base import BaseModel, ModelConfig, ModelRegistry

@ModelRegistry.register("custom")
class CustomModel(BaseModel):
    def generate(self, prompt: str) -> str:
        # Your implementation
        pass
    
    def is_available(self) -> bool:
        return True
```

## Arbiters

| Type | Speed | Accuracy | Cost |
|------|-------|----------|------|
| `heuristic` | ⚡ Fast | Medium | Free |
| `llm` | 🐢 Slow | High | $$ |
| `hybrid` | 🔄 Medium | High | $ |

## Output

Results are saved to `./output/`:
- `graphics/` - PNG visualizations
- `logs/` - JSONL match logs

Visualizations:
- `win_matrix.png` - Head-to-head wins
- `probability_matrix.png` - Win probabilities
- `alpharank.png` - AlphaRank scores
- `elo.png` - Elo ratings
- `dashboard.png` - Summary view
