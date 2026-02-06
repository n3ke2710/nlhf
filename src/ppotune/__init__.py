"""Alias package pointing to rlhf.ppotune."""

import importlib
import sys

_module = importlib.import_module("rlhf.ppotune")

sys.modules[__name__] = _module
