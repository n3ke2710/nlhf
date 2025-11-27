# Quick Command Reference for NLHF

## 🚀 First Run (Full Pipeline)

```bash
# Clone and install
git clone https://github.com/buttercutter/nlhf.git
cd nlhf/experiment

# Set up environment (Python 3.10)
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Run
jupyter notebook nlhf_learning.ipynb
```

**Execute in notebook**: Cells 1-23 in order

---

## ⚡ Quick Start (With Pre-trained Model)

```bash
# If venv was deleted but model remains
cd nlhf/experiment
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Check for model
ls -la qwen2.5-3b-tldr-lora/

# Run
jupyter notebook nlhf_learning.ipynb
```

**Execute in notebook**:
- Cells 1-4 (initialization)
- ❌ SKIP 5-7 (SFT training)
- ✅ Cell 8 (load pre-trained model)
- Cells 9-23 (continue)

---

## 🔧 Installation Verification

```bash
# Library versions
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"

# CUDA and GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0)}')"

# GPU memory
nvidia-smi

# Test imports
python -c "from transformers import Qwen2ForCausalLM, Qwen2ForSequenceClassification; print('✅ Qwen2 imports OK')"
```

---

## 📊 Check Results

```bash
# Check project structure
tree -L 2 -d

# Folder sizes
du -sh qwen2.5-3b-tldr-lora/
du -sh policies/

# Recent logs
tail -50 logs/exp_*.log

# View generations
cat visualizations/sample_generations.txt | head -50
```

---

## 🐛 Diagnostics

### CUDA Out of Memory
```python
# In notebook (cell 5) change:
BATCH_SIZE = 2  # was 4
GRADIENT_ACCUMULATION_STEPS = 8  # was 4

# Or use 8-bit quantization (cell 4):
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)
```

### Check GPU Memory
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Free memory (in Python/notebook)
import torch
torch.cuda.empty_cache()
```

### Reinstall Dependencies
```bash
# Complete reinstall
pip uninstall torch transformers peft -y
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt --force-reinstall
```

---

## 📦 Backup and Restore

### Save Model to Server
```bash
# Archive trained model
tar -czf qwen2.5-3b-tldr-lora.tar.gz qwen2.5-3b-tldr-lora/

# Check size
ls -lh qwen2.5-3b-tldr-lora.tar.gz

# Download from server (on local machine)
scp user@server:/path/to/nlhf/experiment/qwen2.5-3b-tldr-lora.tar.gz .
```

### Restore Model
```bash
# Extract
tar -xzf qwen2.5-3b-tldr-lora.tar.gz

# Verify structure
ls -la qwen2.5-3b-tldr-lora/
```

---

## 🔄 Run on Another Server

```bash
# 1. Transfer model archive
scp qwen2.5-3b-tldr-lora.tar.gz newserver:/path/to/nlhf/experiment/

# 2. On new server
cd /path/to/nlhf/experiment
tar -xzf qwen2.5-3b-tldr-lora.tar.gz

# 3. Set up environment
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Run with pre-trained model (cell 8)
jupyter notebook nlhf_learning.ipynb
```

---

## 📝 Useful Python Snippets

### Quick Model Check (in Python/notebook)
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-3B-Instruct"
lora_path = "./qwen2.5-3b-tldr-lora/checkpoint-500"

# Load
base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda:0")
model = PeftModel.from_pretrained(base, lora_path)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Test
prompt = "SUBREDDIT: r/test\nTITLE: Test\nPOST: Hello world\nTL;DR:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### Clear GPU Memory
```python
import torch
import gc

# Delete model
del model
del base
gc.collect()
torch.cuda.empty_cache()

# Check
print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
```

---

## 🎯 Typical Scenarios

### Scenario 1: Full training from scratch
Cells: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 9 → 10 → 11 → 12 → 13 → 14 → ... → 23

### Scenario 2: Load SFT, train policies
Cells: 1 → 2 → 3 → 4 → **8** (skip 5-7) → 9 → 10 → 11 → 12 → 13 → ... → 23

### Scenario 3: Only generation (everything trained)
Cells: 1 → 2 → 3 → 4 → **8** → skip 9-12 → **14** (load policies) → 15 → ... → 23

---

**Last Updated**: November 2024  
**Version**: 1.0
