# NLHF

Коротко: `src/nashlearninghf` — модульная реализация v3 (турнирное сравнение моделей) с OOP‑архитектурой и тестами. `src/rlhf` — RLHF/PPOTune модуль и PPO‑рецепт.

## Установка (uv)

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Для dev‑зависимостей:

```bash
uv pip install -e ".[dev]"
```

## Быстрый запуск

```bash
# локальный mock‑эксперимент
python scripts/python/exp1.py

# или через bash
bash scripts/bash/exp1.sh
```

## RLHF (PPO)

Конфиги: `scripts/configs/ppo/tldr/`

Запуск (пример):

```bash
bash scripts/bash/ppo/llama_3.2_1b_ppo.sh
```

Документация RLHF и примеры `.env`: `src/rlhf/README.md`.

## Тесты

```bash
pytest -q --cov=src/nashlearninghf --cov-report=term-missing
```

## Документация компонентов

См. `src/README.md` и `src/rlhf/README.md`.
