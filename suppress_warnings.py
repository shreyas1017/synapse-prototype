"""
Suppress known non-critical library warnings for clean console output.
Import this at the top of main.py before any other imports.
"""

import warnings
import os
import logging

# Suppress Python warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress TensorFlow/ONNX logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Suppress HuggingFace warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"     # NEW
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # NEW

# Suppress PyTorch
os.environ["PYTHONWARNINGS"] = "ignore"

# Suppress third-party loggers
for noisy_logger in [
    "ultralytics",
    "easyocr",
    "transformers",
    "torch",
    "PIL",
    "urllib3",
    "huggingface_hub",      # NEW
    "timm",                 # NEW
]:
    logging.getLogger(noisy_logger).setLevel(logging.ERROR)
