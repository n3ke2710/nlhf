"""
Logging Module for NLHF
=======================

Centralized logging configuration.
"""

import logging
import os
from typing import Optional
from .config import NLHFConfig


def setup_logger(
    name: str = 'nlhf',
    config: Optional[NLHFConfig] = None,
    log_file: Optional[str] = None,
    level: str = 'INFO'
) -> logging.Logger:
    """
    Setup logger for NLHF experiments.
    
    Args:
        name: Logger name
        config: NLHF configuration (optional)
        log_file: Path to log file (optional, uses config if not provided)
        level: Logging level
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid adding multiple handlers if re-running
    if logger.handlers:
        return logger
    
    # Determine log file path
    if log_file is None and config is not None:
        os.makedirs(config.exp_dir, exist_ok=True)
        os.makedirs(config.vis_dir, exist_ok=True)
        log_file = config.log_file
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f'Logging to file: {log_file}')
    
    # Reduce noise from transformers
    try:
        from transformers import logging as transformers_logging
        transformers_logging.set_verbosity_error()
    except ImportError:
        pass
    
    return logger


def get_logger(name: str = 'nlhf') -> logging.Logger:
    """Get existing logger by name"""
    return logging.getLogger(name)
