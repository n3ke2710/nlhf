# NLHF (Nash Learning from Human Feedback) Experiment

Implementation of the **Nash Learning from Human Feedback** algorithm for training language models based on game theory and human preferences.

## 📚 Documentation

- 📖 **This file** - complete installation and running guide
- ⚡ **[QUICKSTART.md](QUICKSTART.md)** - command cheat sheet for typical scenarios
- ❓ **[FAQ.md](FAQ.md)** - frequently asked questions and troubleshooting
- 📓 **[nlhf_learning.ipynb](nlhf_learning.ipynb)** - interactive experiment

## 📋 Description

This experiment demonstrates the application of NLHF to the task of generating TL;DR summaries for Reddit posts using the **Qwen2.5-3B-Instruct** model.

### Key Features:
- 🎯 **SFT baseline**: Supervised Fine-Tuning on the `trl-lib/tldr` dataset
- 🎲 **Policy distribution**: Creating N=5 perturbed policies via LoRA perturbations
- 🏆 **Reward model**: Training reward model with KL-regularization
- 📊 **Nash equilibrium**: Analysis of equilibrium in multi-agent game

## 🔧 Requirements

### Minimum System Requirements:
- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with minimum 24GB VRAM (A100 40GB recommended)
- **CUDA**: 11.8 or higher
- **RAM**: 32GB+
- **Disk**: 50GB free space

### Tested Configurations:
✅ Python 3.10 + PyTorch 2.1.2 + CUDA 11.8 + A100 (40GB)  
✅ Python 3.11 + PyTorch 2.1.2 + CUDA 12.1 + A6000 (48GB)

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/buttercutter/nlhf.git
cd nlhf/experiment
```

### 2. Create Virtual Environment (Python 3.10)

#### Linux/macOS:
```bash
# Check Python version
python3.10 --version

# Create virtual environment
python3.10 -m venv venv

# Activate
source venv/bin/activate
```

#### Windows:
```bash
# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CUDA 11.8)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

**Important**: If you have a different CUDA version, select the appropriate PyTorch version at [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Verify Installation

```python
# Run in Python to verify
import torch
import transformers
from peft import LoraConfig

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Transformers: {transformers.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

Expected output:
```
PyTorch: 2.1.2
CUDA available: True
CUDA version: 11.8
Transformers: 4.40.0
GPU: NVIDIA A100-SXM4-40GB
```

## 📓 Running the Experiment

### Option 1: Jupyter Notebook (Recommended)

```bash
# Launch Jupyter
jupyter notebook nlhf_learning.ipynb
```

**Cell Execution Order:**

1. **Cells 1-4**: Initialization, model and data loading
2. **Cells 5-7**: SFT training (or use cell 8 to load pre-trained model)
3. **Cell 8** *(optional)*: Load previously trained SFT model
4. **Cells 9-12**: Create N policies with LoRA perturbations
5. **Cell 13**: Generate responses from all policies
6. **Cells 14-15**: Create preference pairs and train reward model
7. **Cells 16-22**: Visualize results

### Option 2: Using Pre-trained Model (NO retraining) ⚡

**Important**: If you deleted the virtual environment but kept the model folder!

1. **Ensure the model folder exists**:
   ```bash
   ls -la qwen2.5-3b-tldr-lora/
   # Should see: checkpoint-XXX/ or adapter_config.json
   ```

2. **Execute cells 1-4** (initialization, load base model)

3. **SKIP cells 5-7** (SFT training - not needed!)

4. **Run cell 8** - load pre-trained model:
   ```python
   # In cell 8 by default:
   SAVED_MODEL_DIR = "./qwen2.5-3b-tldr-lora"
   
   # If model is elsewhere on server:
   SAVED_MODEL_DIR = "/path/to/your/saved/model"
   ```
   
   Cell automatically:
   - ✅ Finds the latest checkpoint (e.g., `checkpoint-500`)
   - ✅ Loads base Qwen2.5-3B model (if not already loaded)
   - ✅ Applies LoRA adapter
   - ✅ Performs test generation for verification

5. **Continue from cell 9** (create policies)

**Time saved**: ~15-20 minutes (skip SFT training)

### Option 3: Python Script

```bash
# Convert notebook to Python script
jupyter nbconvert --to script nlhf_learning.ipynb

# Run
python nlhf_learning.py
```

## 📂 Project Structure

```
experiment/
├── nlhf_learning.ipynb          # Main notebook (23 cells)
├── requirements.txt              # Dependencies for Python 3.10+
├── README.md                     # This file
│
├── qwen2.5-3b-tldr-lora/        # ✅ Saved SFT model (after cell 7)
│   ├── checkpoint-100/           # Intermediate checkpoints
│   ├── checkpoint-200/
│   ├── checkpoint-500/          # Final checkpoint
│   └── training_args.bin
│
├── policies/                     # ✅ Policies (after cells 9-12)
│   ├── policy_0/                # LoRA adapter for policy 0
│   ├── policy_1/                # LoRA adapter for policy 1
│   ├── policy_2/
│   ├── policy_3/
│   └── policy_4/
│
├── logs/                         # ✅ Training logs
│   └── exp_20251127_*.log       # Timestamp-based logs
│
└── visualizations/               # ✅ Results (after cell 14+)
    ├── sample_generations.txt   # Generation examples from all policies
    ├── reward_distribution.png  # Reward distribution
    └── nash_convergence.png     # Convergence analysis
```

**Important**: Folders `qwen2.5-3b-tldr-lora/`, `policies/`, `logs/`, `visualizations/` are created automatically when executing corresponding cells.

## 🔄 Rerunning After Deleting venv

**Scenario**: You accidentally deleted the virtual environment on the server, but the model folders remain.

### What's Preserved:
- ✅ `qwen2.5-3b-tldr-lora/` - trained SFT model (~3GB)
- ✅ `policies/` - if you created policies
- ✅ All results in `visualizations/`

### Quick Recovery:

```bash
# 1. Recreate venv
python3.10 -m venv venv
source venv/bin/activate

# 2. Reinstall dependencies
pip install --upgrade pip
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook nlhf_learning.ipynb
```

### In notebook:
1. Execute cells **1-4** (initialization)
2. **SKIP cells 5-7** ❌ (no need to retrain!)
3. Execute **cell 8** ✅ (loads pre-trained model from `qwen2.5-3b-tldr-lora/`)
4. Continue from cell 9

**Savings**: ~20 minutes training + ~10GB dataset download traffic

## ⚙️ Configuration

### Main Hyperparameters (in notebook):

```python
# SFT Training
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 1e-5
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch_size = 16
LORA_R = 32
LORA_ALPHA = 64

# NLHF Parameters
N_POLICIES = 5                    # Number of policies
PERTURBATION_SCALE = 0.1          # Perturbation scale for policies
KL_COEFFICIENT = 0.1              # τ in KL-regularization

# Generation
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.8
TOP_P = 0.9
REPETITION_PENALTY = 1.2
```

### Adapting to Different GPUs:

**A100 (40GB)** - recommended configuration (default)
```python
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
```

**RTX 3090/4090 (24GB)**:
```python
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
# Add gradient_checkpointing_enable() in cell 5
```

**V100 (16GB)** - minimum configuration:
```python
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
# Use 8-bit quantization (load_in_8bit=True)
```

## 🐛 Troubleshooting

### Issue: `ImportError: cannot import 'albert'`
**Solution**: Code already contains workaround - uses `Qwen2ForSequenceClassification` directly instead of `AutoModelForSequenceClassification`.

### Issue: CUDA Out of Memory
**Solutions**:
1. Reduce `BATCH_SIZE` to 1-2
2. Increase `GRADIENT_ACCUMULATION_STEPS`
3. Enable gradient checkpointing (already enabled in code)
4. Use 8-bit quantization:
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       load_in_8bit=True,
       device_map="auto"
   )
   ```

### Issue: Garbage text generation ("zro zro zro...")
**Solution**: Code contains improved generation parameters and garbage detection:
- `repetition_penalty=1.2`
- `no_repeat_ngram_size=3`
- `top_p=0.9`, `top_k=50`
- Automatic filtering via `is_garbage()`

### Issue: Slow generation
**Solution**:
```python
# Reduce number of validation samples
val_prompts = val_prompts[:50]  # Instead of 100

# Increase batch_size for generation
GENERATION_BATCH_SIZE = 16  # If VRAM allows
```

## 📊 Expected Results

After successful execution you will get:

1. **Trained SFT model**: in `qwen2.5-3b-tldr-lora/`
2. **5 policies**: in `policies/policy_0/` ... `policy_4/`
3. **Reward model**: trained on preference pairs
4. **Visualizations**:
   - Reward distribution across policies
   - Generation examples from baseline and each policy
   - Quality metrics (garbage ratio, diversity)
5. **Training logs**: in `logs/exp_*.log`

### Approximate Execution Time (A100):
- SFT Training (500 steps): ~15-20 minutes
- Policy creation (5 policies): ~2-3 minutes
- Response generation (100 prompts × 6 models): ~10-15 minutes
- Reward model training: ~5-10 minutes
- **Total**: ~40-50 minutes

## 📚 Additional Resources

### Project Documentation:
- 📖 **[QUICKSTART.md](QUICKSTART.md)** - commands for quick start and typical scenarios
- ❓ **[FAQ.md](FAQ.md)** - frequently asked questions, troubleshooting, tips
- 📓 **[nlhf_learning.ipynb](nlhf_learning.ipynb)** - main interactive notebook

### External Resources:
- [PEFT Documentation](https://huggingface.co/docs/peft) - Parameter-Efficient Fine-Tuning
- [TRL Documentation](https://huggingface.co/docs/trl) - Transformer Reinforcement Learning
- [Qwen2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) - base model
- [PyTorch Installation](https://pytorch.org/get-started/locally/) - installation for different CUDA versions

## 🤝 Support and Contact

If you have questions or problems:
1. 📖 Check **[FAQ.md](FAQ.md)** - the answer is likely there
2. 🔍 Check logs in `logs/exp_*.log`
3. 🐛 Open an [Issue on GitHub](https://github.com/buttercutter/nlhf/issues)
4. 💬 Include system information: Python version, GPU model, CUDA version, error traceback

## 📝 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- Hugging Face for Transformers and PEFT libraries
- Qwen Team for the open-source Qwen2.5 model
- TRL Team for dataset and RLHF utilities

---
**Version**: 1.0  
**Last Updated**: November 2024  
**Author**: [Your Name]
