# NashLearningHF (src)

Этот пакет содержит модульную OOP‑реализацию эксперимента v3 (турнирное сравнение моделей) с разделением на доменные компоненты. Цель — обеспечить чистую архитектуру, тестируемость и удобство расширения.

## Состав компонентов

### `nashlearninghf/`
Корневой пакет с экспортами ключевых классов и фабрик.

### `nashlearninghf/models/`
Абстракции и реализации моделей для генерации TL;DR.

- `base.py`
  - `ModelConfig` — конфигурация модели (api/local/mock/lora и т.д.).
  - `BaseModel` — абстрактный интерфейс `generate()` и `is_available()`.
  - `ModelRegistry` — реестр типов моделей и фабрика создания по конфигу.
  - `create_response_fn()` — обёртка для использования модели в турнире.

- `mock.py`
  - `MockModel` — детерминированные ответы для локальных запусков без API.
  - `create_mock_models()` — удобное создание набора mock‑моделей.

- `api.py`
  - `APIModel` — базовый OpenAI‑совместимый клиент.
  - `OpenAIModel`, `DeepInfraModel`, `AnthropicModel` — конкретные провайдеры.
  - `create_openai_models()`, `create_deepinfra_models()` — фабрики наборов.

- `local.py`
  - `LocalModel` — локальная HF‑модель.
  - `LoRAModel` — LoRA‑адаптер на базе HF/PEFT.
  - `load_local_model()`, `load_lora_model()` — удобные загрузчики.

### `nashlearninghf/arbiters/`
Механизмы сравнения ответов (арбитры).

- `base.py`
  - `ArbiterResult` — результат сравнения.
  - `BaseArbiter` — интерфейс `compare()`.

- `heuristic.py`
  - `HeuristicArbiter` — быстрые эвристики (сжатие/пересечения/читаемость).

- `llm.py`
  - `LLMArbiter` — сравнение с помощью LLM через API.

- `hybrid.py`
  - `HybridArbiter` — комбинирует эвристику и LLM.
  - `create_arbiter()` — фабрика арбитров по типу.

### `nashlearninghf/tournament.py`
Ядро турнира для сравнения моделей.

- `Tournament` — матчинг, матрицы побед, AlphaRank/Elo‑метрики.
- `MatchResult`, `TournamentStats` — структуры результатов.
- `heuristic_arbiter()` — функция‑арбитр на базе `score_summary()`.
- `llm_arbiter_factory()` и `remote_arbiter_factory()` — фабрики LLM‑арбитров.

### `nashlearninghf/scoring.py`
- `score_summary()` — улучшенная эвристика качества TL;DR.
- `create_better_preference_pairs()` — формирование preference pairs.

### `nashlearninghf/pipeline.py`
Высокоуровневый запуск эксперимента.

- `PromptProvider` — абстракция источника промптов.
- `InMemoryPromptProvider` — простой провайдер.
- `TournamentExperiment` — оркестратор запуска турнира.

### `nashlearninghf/config/`
- `experiment.py` — `ExperimentConfig` (параметры запуска, директории).
- `models.py` — `ModelsConfig` + `load_models_from_yaml()`.

### `nashlearninghf/data/`
Загрузка данных (HuggingFace datasets).

- `load_sft_dataset()` — SFT датасет.
- `load_validation_prompts()` — промпты для валидации.
- `load_real_preferences()` — реальные пары предпочтений.

## Быстрый локальный запуск (без API)

```bash
python ../scripts/python/exp1.py
```

## Тесты

```bash
pytest -q --cov=src/nashlearninghf --cov-report=term-missing
```

---

## RLHF (TunePPO)

Отдельный модуль PPO‑оптимизации находится в `src/rlhf`.
Подробности и примеры запуска см. в `src/rlhf/README.md`.
