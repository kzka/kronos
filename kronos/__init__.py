import logging
import sys


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

__all__ = [
    "datasets",
    "evaluators",
    "losses",
    "models",
    "trainers",
    "utils",
]
