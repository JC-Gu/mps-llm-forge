"""
Llama 3 8B LoRA Fine-tuning Package

This package provides utilities for fine-tuning Llama 3 models using LoRA
on Apple Silicon Mac Minis with optimized memory management and performance.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .model_utils import load_model_and_tokenizer, setup_lora
from .data_processing import DataProcessor, load_and_process_data
from .training import Trainer
from .inference import InferenceEngine

__all__ = [
    "load_model_and_tokenizer",
    "setup_lora", 
    "DataProcessor",
    "load_and_process_data",
    "Trainer",
    "InferenceEngine"
]
