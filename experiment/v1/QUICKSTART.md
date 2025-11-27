# Шпаргалка по быстрым командам NLHF

## 🚀 Первый запуск (полный цикл)

```bash
# Клонирование и установка
git clone https://github.com/buttercutter/nlhf.git
cd nlhf/experiment

# Настройка окружения (Python 3.10)
python3.10 -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install --upgrade pip
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Проверка CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Запуск
jupyter notebook nlhf_learning.ipynb
```

**Выполнить в notebook**: Ячейки 1-23 по порядку

---

## ⚡ Быстрый запуск (с готовой моделью)

```bash
# Если venv удалён, но модель осталась
cd nlhf/experiment
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Проверка наличия модели
ls -la qwen2.5-3b-tldr-lora/

# Запуск
jupyter notebook nlhf_learning.ipynb
```

**Выполнить в notebook**:
- Ячейки 1-4 (инициализация)
- ❌ ПРОПУСТИТЬ 5-7 (обучение SFT)
- ✅ Ячейка 8 (загрузка готовой модели)
- Ячейки 9-23 (продолжение)

---

## 🔧 Проверка установки

```bash
# Версии библиотек
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"

# CUDA и GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0)}')"

# Память GPU
nvidia-smi

# Тест импортов
python -c "from transformers import Qwen2ForCausalLM, Qwen2ForSequenceClassification; print('✅ Qwen2 imports OK')"
```

---

## 📊 Проверка результатов

```bash
# Проверка структуры проекта
tree -L 2 -d

# Размеры папок
du -sh qwen2.5-3b-tldr-lora/
du -sh policies/

# Последние логи
tail -50 logs/exp_*.log

# Просмотр генераций
cat visualizations/sample_generations.txt | head -50
```

---

## 🐛 Диагностика проблем

### CUDA Out of Memory
```python
# В notebook (ячейка 5) измените:
BATCH_SIZE = 2  # было 4
GRADIENT_ACCUMULATION_STEPS = 8  # было 4

# Или используйте 8-bit quantization (ячейка 4):
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)
```

### Проверка GPU памяти
```bash
# Мониторинг в реальном времени
watch -n 1 nvidia-smi

# Освобождение памяти (в Python/notebook)
import torch
torch.cuda.empty_cache()
```

### Переустановка зависимостей
```bash
# Полная переустановка
pip uninstall torch transformers peft -y
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt --force-reinstall
```

---

## 📦 Бэкап и восстановление

### Сохранение модели на сервер
```bash
# Архивирование обученной модели
tar -czf qwen2.5-3b-tldr-lora.tar.gz qwen2.5-3b-tldr-lora/

# Проверка размера
ls -lh qwen2.5-3b-tldr-lora.tar.gz

# Скачивание с сервера (на локальный компьютер)
scp user@server:/path/to/nlhf/experiment/qwen2.5-3b-tldr-lora.tar.gz .
```

### Восстановление модели
```bash
# Распаковка
tar -xzf qwen2.5-3b-tldr-lora.tar.gz

# Проверка структуры
ls -la qwen2.5-3b-tldr-lora/
```

---

## 🔄 Запуск на другом сервере

```bash
# 1. Перенести архив модели
scp qwen2.5-3b-tldr-lora.tar.gz newserver:/path/to/nlhf/experiment/

# 2. На новом сервере
cd /path/to/nlhf/experiment
tar -xzf qwen2.5-3b-tldr-lora.tar.gz

# 3. Настроить окружение
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Запустить с готовой моделью (ячейка 8)
jupyter notebook nlhf_learning.ipynb
```

---

## 📝 Полезные Python snippets

### Быстрая проверка модели (в Python/notebook)
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-3B-Instruct"
lora_path = "./qwen2.5-3b-tldr-lora/checkpoint-500"

# Загрузка
base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda:0")
model = PeftModel.from_pretrained(base, lora_path)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Тест
prompt = "SUBREDDIT: r/test\nTITLE: Test\nPOST: Hello world\nTL;DR:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### Очистка памяти GPU
```python
import torch
import gc

# Удаление модели
del model
del base
gc.collect()
torch.cuda.empty_cache()

# Проверка
print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
```

---

## 🎯 Типичные сценарии

### Сценарий 1: Полное обучение с нуля
Ячейки: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 9 → 10 → 11 → 12 → 13 → 14 → ... → 23

### Сценарий 2: Загрузка SFT, обучение политик
Ячейки: 1 → 2 → 3 → 4 → **8** (пропуск 5-7) → 9 → 10 → 11 → 12 → 13 → ... → 23

### Сценарий 3: Только генерация (всё уже обучено)
Ячейки: 1 → 2 → 3 → 4 → **8** → пропуск 9-12 → **14** (загрузка политик) → 15 → ... → 23

---

**Дата обновления**: Ноябрь 2024  
**Версия**: 1.0
