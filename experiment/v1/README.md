# NLHF (Nash Learning from Human Feedback) Experiment

Реализация алгоритма **Nash Learning from Human Feedback** для обучения языковых моделей на основе теории игр и человеческих предпочтений.

## � Документация

- 📖 **Этот файл** - полное руководство по установке и запуску
- ⚡ **[QUICKSTART.md](QUICKSTART.md)** - шпаргалка с командами для типичных сценариев
- ❓ **[FAQ.md](FAQ.md)** - часто задаваемые вопросы и решения проблем
- 📓 **[nlhf_learning.ipynb](nlhf_learning.ipynb)** - интерактивный эксперимент

## �📋 Описание

Этот эксперимент демонстрирует применение NLHF к задаче генерации TL;DR саммари для постов Reddit с использованием модели **Qwen2.5-3B-Instruct**.

### Ключевые особенности:
- 🎯 **SFT baseline**: Supervised Fine-Tuning на датасете `trl-lib/tldr`
- 🎲 **Policy distribution**: Создание N=5 возмущённых политик через LoRA-возмущения
- 🏆 **Reward model**: Обучение reward model с KL-регуляризацией
- 📊 **Nash equilibrium**: Анализ равновесия в многоагентной игре

## 🔧 Требования

### Минимальные системные требования:
- **Python**: 3.10 или выше
- **GPU**: NVIDIA GPU с минимум 24GB VRAM (рекомендуется A100 40GB)
- **CUDA**: 11.8 или выше
- **RAM**: 32GB+
- **Disk**: 50GB свободного места

### Проверенные конфигурации:
✅ Python 3.10 + PyTorch 2.1.2 + CUDA 11.8 + A100 (40GB)  
✅ Python 3.11 + PyTorch 2.1.2 + CUDA 12.1 + A6000 (48GB)

## 🚀 Быстрый старт

### 1. Клонирование репозитория
```bash
git clone https://github.com/buttercutter/nlhf.git
cd nlhf/experiment
```

### 2. Создание виртуального окружения (Python 3.10)

#### Linux/macOS:
```bash
# Проверка версии Python
python3.10 --version

# Создание виртуального окружения
python3.10 -m venv venv

# Активация
source venv/bin/activate
```

#### Windows:
```bash
# Создание виртуального окружения
python -m venv venv

# Активация
venv\Scripts\activate
```

### 3. Установка зависимостей

```bash
# Обновление pip
pip install --upgrade pip

# Установка PyTorch (CUDA 11.8)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Установка остальных зависимостей
pip install -r requirements.txt
```

**Важно**: Если у вас другая версия CUDA, выберите соответствующую версию PyTorch на [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Проверка установки

```python
# Запустите в Python для проверки
import torch
import transformers
from peft import LoraConfig

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Transformers: {transformers.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

Ожидаемый вывод:
```
PyTorch: 2.1.2
CUDA available: True
CUDA version: 11.8
Transformers: 4.40.0
GPU: NVIDIA A100-SXM4-40GB
```

## 📓 Запуск эксперимента

### Вариант 1: Jupyter Notebook (рекомендуется)

```bash
# Запуск Jupyter
jupyter notebook nlhf_learning.ipynb
```

**Порядок выполнения ячеек:**

1. **Ячейки 1-4**: Инициализация, загрузка модели и данных
2. **Ячейки 5-7**: SFT обучение (или используйте ячейку 8 для загрузки готовой модели)
3. **Ячейка 8** *(опционально)*: Загрузка ранее обученной SFT модели
4. **Ячейки 9-12**: Создание N политик с LoRA-возмущениями
5. **Ячейка 13**: Генерация ответов от всех политик
6. **Ячейки 14-15**: Создание preference pairs и обучение reward model
7. **Ячейки 16-22**: Визуализация результатов

### Вариант 2: Использование готовой модели (БЕЗ переобучения) ⚡

**Важно**: Если вы удалили виртуальное окружение, но у вас сохранилась папка с моделью!

1. **Убедитесь, что папка с моделью существует**:
   ```bash
   ls -la qwen2.5-3b-tldr-lora/
   # Должны быть: checkpoint-XXX/ или adapter_config.json
   ```

2. **Выполните ячейки 1-4** (инициализация, загрузка базовой модели)

3. **ПРОПУСТИТЕ ячейки 5-7** (обучение SFT - не нужно!)

4. **Запустите ячейку 8** - загрузка готовой модели:
   ```python
   # В ячейке 8 по умолчанию:
   SAVED_MODEL_DIR = "./qwen2.5-3b-tldr-lora"
   
   # Если модель на сервере в другом месте:
   SAVED_MODEL_DIR = "/path/to/your/saved/model"
   ```
   
   Ячейка автоматически:
   - ✅ Найдёт последний checkpoint (например, `checkpoint-500`)
   - ✅ Загрузит базовую модель Qwen2.5-3B (если ещё не загружена)
   - ✅ Применит LoRA адаптер
   - ✅ Сделает тестовую генерацию для проверки

5. **Продолжите с ячейки 9** (создание политик)

**Экономия времени**: ~15-20 минут (пропуск SFT обучения)

### Вариант 3: Python скрипт

```bash
# Конвертация notebook в Python скрипт
jupyter nbconvert --to script nlhf_learning.ipynb

# Запуск
python nlhf_learning.py
```

## 📂 Структура проекта

```
experiment/
├── nlhf_learning.ipynb          # Основной notebook (23 ячейки)
├── requirements.txt              # Зависимости для Python 3.10+
├── README.md                     # Этот файл
│
├── qwen2.5-3b-tldr-lora/        # ✅ Сохранённая SFT модель (после ячейки 7)
│   ├── checkpoint-100/           # Промежуточные чекпоинты
│   ├── checkpoint-200/
│   ├── checkpoint-500/          # Финальный чекпоинт
│   └── training_args.bin
│
├── policies/                     # ✅ Политики (после ячеек 9-12)
│   ├── policy_0/                # LoRA адаптер для политики 0
│   ├── policy_1/                # LoRA адаптер для политики 1
│   ├── policy_2/
│   ├── policy_3/
│   └── policy_4/
│
├── logs/                         # ✅ Логи обучения
│   └── exp_20251127_*.log       # Timestamp-based логи
│
└── visualizations/               # ✅ Результаты (после ячейки 14+)
    ├── sample_generations.txt   # Примеры генерации всех политик
    ├── reward_distribution.png  # Распределение rewards
    └── nash_convergence.png     # Анализ сходимости
```

**Важно**: Папки `qwen2.5-3b-tldr-lora/`, `policies/`, `logs/`, `visualizations/` создаются автоматически при выполнении соответствующих ячеек.

## 🔄 Повторный запуск после удаления venv

**Сценарий**: Вы случайно удалили виртуальное окружение на сервере, но папки с моделями остались.

### Что сохранилось:
- ✅ `qwen2.5-3b-tldr-lora/` - обученная SFT модель (~3GB)
- ✅ `policies/` - если вы успели создать политики
- ✅ Все результаты в `visualizations/`

### Быстрое восстановление:

```bash
# 1. Пересоздать venv
python3.10 -m venv venv
source venv/bin/activate

# 2. Переустановить зависимости
pip install --upgrade pip
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 3. Запустить Jupyter
jupyter notebook nlhf_learning.ipynb
```

### В notebook:
1. Выполните ячейки **1-4** (инициализация)
2. **ПРОПУСТИТЕ ячейки 5-7** ❌ (не нужно переобучать!)
3. Выполните **ячейку 8** ✅ (загрузит готовую модель из `qwen2.5-3b-tldr-lora/`)
4. Продолжите работу с ячейки 9

**Экономия**: ~20 минут обучения + ~10GB трафика скачивания датасета

## ⚙️ Конфигурация

### Основные гиперпараметры (в notebook):

```python
# SFT Training
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 1e-5
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4  # Эффективный batch_size = 16
LORA_R = 32
LORA_ALPHA = 64

# NLHF Parameters
N_POLICIES = 5                    # Количество политик
PERTURBATION_SCALE = 0.1          # Масштаб возмущения для политик
KL_COEFFICIENT = 0.1              # τ в KL-регуляризации

# Generation
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.8
TOP_P = 0.9
REPETITION_PENALTY = 1.2
```

### Настройка под разные GPU:

**A100 (40GB)** - рекомендуемая конфигурация (по умолчанию)
```python
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
```

**RTX 3090/4090 (24GB)**:
```python
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
# Добавьте gradient_checkpointing_enable() в ячейке 5
```

**V100 (16GB)** - минимальная конфигурация:
```python
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
# Используйте 8-bit quantization (load_in_8bit=True)
```

## 🐛 Устранение неполадок

### Проблема: `ImportError: cannot import 'albert'`
**Решение**: Код уже содержит workaround - использует `Qwen2ForSequenceClassification` напрямую вместо `AutoModelForSequenceClassification`.

### Проблема: CUDA Out of Memory
**Решения**:
1. Уменьшите `BATCH_SIZE` до 1-2
2. Увеличьте `GRADIENT_ACCUMULATION_STEPS`
3. Включите gradient checkpointing (уже включено в коде)
4. Используйте квантизацию 8-bit:
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       load_in_8bit=True,
       device_map="auto"
   )
   ```

### Проблема: Генерация мусорного текста ("zro zro zro...")
**Решение**: Код содержит улучшенные параметры генерации и garbage detection:
- `repetition_penalty=1.2`
- `no_repeat_ngram_size=3`
- `top_p=0.9`, `top_k=50`
- Автоматическая фильтрация через `is_garbage()`

### Проблема: Медленная генерация
**Решение**:
```python
# Уменьшите количество validation samples
val_prompts = val_prompts[:50]  # Вместо 100

# Увеличьте batch_size для генерации
GENERATION_BATCH_SIZE = 16  # Если позволяет VRAM
```

## 📊 Ожидаемые результаты

После успешного запуска вы получите:

1. **Обученная SFT модель**: в `qwen2.5-3b-tldr-lora/`
2. **5 политик**: в `policies/policy_0/` ... `policy_4/`
3. **Reward model**: обученная на preference pairs
4. **Визуализации**:
   - Распределение rewards по политикам
   - Примеры генерации от baseline и каждой политики
   - Метрики качества (garbage ratio, diversity)
5. **Логи обучения**: в `logs/exp_*.log`

### Примерное время выполнения (A100):
- SFT Training (500 steps): ~15-20 минут
- Policy creation (5 policies): ~2-3 минуты
- Response generation (100 prompts × 6 models): ~10-15 минут
- Reward model training: ~5-10 минут
- **Итого**: ~40-50 минут

## 📚 Дополнительные ресурсы

### Документация проекта:
- 📖 **[QUICKSTART.md](QUICKSTART.md)** - команды для быстрого старта и типичных сценариев
- ❓ **[FAQ.md](FAQ.md)** - часто задаваемые вопросы, решение проблем, советы
- 📓 **[nlhf_learning.ipynb](nlhf_learning.ipynb)** - основной интерактивный notebook

### Внешние ресурсы:
- [Документация PEFT](https://huggingface.co/docs/peft) - Parameter-Efficient Fine-Tuning
- [Документация TRL](https://huggingface.co/docs/trl) - Transformer Reinforcement Learning
- [Qwen2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) - базовая модель
- [PyTorch Installation](https://pytorch.org/get-started/locally/) - установка под разные CUDA версии

## 🤝 Контакты и поддержка

Если у вас возникли вопросы или проблемы:
1. 📖 Проверьте **[FAQ.md](FAQ.md)** - вероятно, ответ уже там
2. 🔍 Проверьте логи в `logs/exp_*.log`
3. 🐛 Откройте [Issue на GitHub](https://github.com/buttercutter/nlhf/issues)
4. 💬 Приложите информацию о системе: Python версия, GPU модель, CUDA версия, traceback ошибки

## 📝 Лицензия

MIT License - см. файл LICENSE

## 🙏 Благодарности

- Hugging Face за библиотеки Transformers и PEFT
- Qwen Team за открытую модель Qwen2.5
- TRL Team за датасет и утилиты для RLHF

---
**Версия**: 1.0  
**Дата обновления**: Ноябрь 2024  
**Автор**: [Your Name]
