# RLHF (TunePPO) Module

Этот модуль — локальная копия библиотеки TunePPO (пакет `ppotune`) и рецепт PPO для запуска RLHF‑оптимизации. Код находится в `src/rlhf/ppotune` и `src/rlhf/recipes/ppo.py`.

## Где лежат конфиги

Конфиги запуска (TL;DR) находятся в `scripts/configs/ppo/tldr/`.

Каждый YAML использует параметр `prefix`:

```yaml
prefix: ${oc.env:PPO_PREFIX, .}
```

Это означает, что модели/чекпойнты/логи будут писаться в `${PPO_PREFIX}/models`, `${PPO_PREFIX}/checkpoints`, `${PPO_PREFIX}/logs` и т.д.

## Переменные окружения (.env)

Минимально необходимые переменные зависят от конфигурации:

Обязательные почти всегда:
- `HF_TOKEN` или `HUGGINGFACE_HUB_TOKEN` — доступ к моделям/датасетам HF

Условные:
- `WANDB_API_KEY` — если используете WandB‑логирование (иначе поставьте `WANDB_MODE=offline`)
- `OPENAI_API_KEY` — если в конфиге арбитр использует OpenAI/DeepInfra (см. `evaluator.arbiter.base_url`)

Рекомендуемые для локальных кэшей:
- `HF_HOME`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE`, `TORCH_HOME`, `WANDB_DIR`

Пример заполнения `.env`:

```bash
# ключи
OPENAI_API_KEY=sk-...
DEEPINFRA_API_KEY=...
ANTHROPIC_API_KEY=...
WANDB_API_KEY=...
HF_TOKEN=...
HUGGINGFACE_HUB_TOKEN=...

# локальные директории для кэшей
HF_HOME=./data/hf
HF_DATASETS_CACHE=./data/datasets
TRANSFORMERS_CACHE=./models/hf
TORCH_HOME=./models/torch
WANDB_DIR=./logs/wandb

# префикс для моделей/чекпойнтов PPO
PPO_PREFIX=.

# если нет WANDB_API_KEY
WANDB_MODE=offline
```

## Как запустить

Под каждый YAML есть bash‑скрипт:

```bash
# пример
bash scripts/bash/ppo/llama_3.2_1b_ppo.sh
```

Скрипт:
- подхватывает `.env`
- предупреждает, если нет нужных переменных
- запускает `tune run` с нужным конфигом

Можно переопределить количество процессов:

```bash
NPROC=2 bash scripts/bash/ppo/llama_3.2_1b_ppo.sh
```

## Заполнение конфигов

Минимально нужно указать/проверить:
- `prefix` (или `PPO_PREFIX` в env)
- пути до моделей и reward‑моделей в блоках `policy.ckpt` и `advantage.reward.scorer.ckpt`
- `base_url` и `model` в `evaluator.arbiter` если используете remote‑арбитра

Все остальные параметры уже зафиксированы в `scripts/configs/ppo/tldr/*.yaml` и могут быть изменены при необходимости.
