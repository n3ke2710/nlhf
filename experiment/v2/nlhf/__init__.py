"""
NLHF (Nash Learning from Human Feedback) Package
=================================================

Модульная реализация алгоритма NLHF для обучения языковых моделей
с использованием человеческой обратной связи и KL-регуляризации.

Структура:
- config.py: Конфигурация и константы
- logger.py: Настройка логирования
- data/: Работа с датасетами
- models/: Модели (SFT, Reward Model)
- training/: Тренеры и функции обучения
- evaluation/: Метрики и оценка
- utils/: Вспомогательные функции
"""

__version__ = "2.0.0"
__author__ = "NLHF Team"

from .config import NLHFConfig
from .logger import setup_logger

__all__ = [
    "NLHFConfig",
    "setup_logger",
]
