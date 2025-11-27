# NLHF: Nash Learning from Human Feedback

Implementation of the **Nash Learning from Human Feedback** algorithm for training language models based on game theory and human preferences.

## 📖 About the Project

NLHF is a novel approach to training language models with human preferences, based on the concept of Nash equilibrium from game theory. Unlike traditional RLHF (Reinforcement Learning from Human Feedback), NLHF creates a distribution of policies and trains a reward model with KL-divergence regularization.

## 🗂️ Repository Structure

```
nlhf/
├── experiment/                  # Main experiment
│   ├── nlhf_learning.ipynb     # Jupyter notebook with full pipeline (23 cells)
│   ├── requirements.txt         # Dependencies for Python 3.10+
│   ├── README.md               # Complete installation and running guide
│   ├── QUICKSTART.md           # Cheat sheet with commands for typical scenarios
│   ├── FAQ.md                  # Frequently asked questions and solutions
│   └── .gitignore              # Git ignore for large files
└── README.md                    # This file
```

## 🚀 Быстрый старт

```bash
# Клонирование репозитория
git clone https://github.com/buttercutter/nlhf.git
cd nlhf/experiment

# Установка зависимостей (Python 3.10+)
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Запуск эксперимента
jupyter notebook nlhf_learning.ipynb
```

**Полная документация**: см. [experiment/README.md](experiment/README.md)

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

- 📖 **[experiment/README.md](experiment/README.md)** - полное руководство по установке и запуску
- ⚡ **[experiment/QUICKSTART.md](experiment/QUICKSTART.md)** - команды для быстрого старта
- ❓ **[experiment/FAQ.md](experiment/FAQ.md)** - часто задаваемые вопросы
- 📓 **[experiment/nlhf_learning.ipynb](experiment/nlhf_learning.ipynb)** - интерактивный эксперимент

## 🤝 Участие в разработке

Приветствуются Issues и Pull Requests!

## 📝 Лицензия

MIT License

## 🙏 Благодарности

- [Hugging Face](https://huggingface.co/) за экосистему Transformers
- [Qwen Team](https://github.com/QwenLM/Qwen) за открытую модель
- [TRL](https://github.com/huggingface/trl) за инструменты RLHF