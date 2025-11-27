# TL;DR

**NLHF** — это фреймворк для обучения языковых моделей с помощью Nash Learning from Human Feedback (NLHF), позволяющий генерировать качественные TL;DR-суммаризации для Reddit-постов и других текстов на основе человеческих предпочтений и теории игр.

- 📌 Быстрый старт: обучение, генерация и оценка TL;DR для реальных постов.
- 🚀 Поддержка запуска на GPU (A100 40GB рекомендуется).
- 📊 Визуализация качества и анализ Nash-равновесия.
- 📝 Пример вывода TL;DR: см. файл `sample_generations.txt` в корне репозитория.

---

# NLHF: Nash Learning from Human Feedback

Implementation of the **Nash Learning from Human Feedback** algorithm for training language models based on game theory and human preferences.

## 📖 About the Project

NLHF is a novel approach to training language models with human preferences, based on the concept of Nash equilibrium from game theory. Unlike traditional RLHF (Reinforcement Learning from Human Feedback), NLHF creates a distribution of policies and trains a reward model with KL-divergence regularization.

nlhf/
## 🗂️ Структура репозитория

```
nlhf/
├── experiment/
│   ├── v1/                # Старый pipeline: ноутбук, гайды, примеры
│   │   ├── nlhf_learning.ipynb   # Jupyter notebook: полный pipeline (23+ cells)
│   │   ├── README.md, QUICKSTART.md, FAQ.md
│   │   └── ...
│   └── v2/
│       └── nlhf/          # Модульная реализация NLHF (рекомендуется)
│           ├── config.py      # Конфигурация экспериментов
│           ├── data/         # Загрузка и препроцессинг датасетов
│           ├── logger.py     # Логирование
│           ├── ...           # (models/, training/, evaluation/, utils/ — структура под расширение)
└── README.md                # Этот файл
```

git clone https://github.com/buttercutter/nlhf.git

## 🚀 Быстрый старт

```bash
# Клонирование репозитория
git clone https://github.com/buttercutter/nlhf.git
cd nlhf/experiment/v2

# Установка зависимостей (Python 3.10+)
python3.10 -m venv venv
source venv/bin/activate
pip install -r ../v1/requirements.txt

# Запуск экспериментов (v2)
# (пример: запуск скриптов или модулей из nlhf/)
# python -m nlhf.<module> ...

# Для интерактивного эксперимента (v1):
cd ../v1
jupyter notebook nlhf_learning.ipynb
```

**Документация по v2**: в процессе. Для ознакомления с принципами работы используйте [experiment/v1/README.md](experiment/v1/README.md)

## 📋 Требования

- Python 3.10+
- NVIDIA GPU с минимум 24GB VRAM (рекомендуется A100 40GB)
- CUDA 11.8+
- 32GB+ RAM

## 🎯 Ключевые возможности

- ✅ **SFT Baseline**: Fine-tuning Qwen2.5-3B на датасете TL;DR
- ✅ **Policy Distribution**: Создание N политик через LoRA-возмущения
- ✅ **Reward Model**: Обучение с KL-регуляризацией
- ✅ **Визуализация**: Анализ Nash равновесия и качества генерации
- ✅ **Загрузка готовых моделей**: Возможность пропустить обучение


## 📚 Документация

- 📖 **[experiment/v1/README.md](experiment/v1/README.md)** — полное руководство по установке и запуску (v1)
- ⚡ **[experiment/v1/QUICKSTART.md](experiment/v1/QUICKSTART.md)** — команды для быстрого старта (v1)
- ❓ **[experiment/v1/FAQ.md](experiment/v1/FAQ.md)** — часто задаваемые вопросы (v1)
- 📓 **[experiment/v1/nlhf_learning.ipynb](experiment/v1/nlhf_learning.ipynb)** — интерактивный эксперимент (v1)
- 🛠️ **experiment/v2/nlhf/** — модульная реализация NLHF (см. docstring и структуру модулей)

## 🤝 Участие в разработке

Приветствуются Issues и Pull Requests!

## 📝 Лицензия

MIT License

## 🙏 Благодарности

- [Hugging Face](https://huggingface.co/) за экосистему Transformers
- [Qwen Team](https://github.com/QwenLM/Qwen) за открытую модель
- [TRL](https://github.com/huggingface/trl) за инструменты RLHF