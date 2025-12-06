"""
Training module for Llama 3 LoRA fine-tuning on Mac Mini.

This module provides the main training loop with optimizations
for Apple Silicon hardware and memory management.
"""

import os
import logging
import time
import psutil
from typing import Dict, Any, Optional
from pathlib import Path
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import DatasetDict
import yaml
try:
    from .model_utils import (
        load_model_and_tokenizer,
        setup_lora,
        optimize_for_mac_mini,
        get_training_arguments,
        print_model_info,
        get_device_info
    )
    from .data_processing import load_and_process_data
except ImportError:
    # When running directly, use absolute imports
    from model_utils import (
        load_model_and_tokenizer,
        setup_lora,
        optimize_for_mac_mini,
        get_training_arguments,
        print_model_info,
        get_device_info
    )
    from data_processing import load_and_process_data

logger = logging.getLogger(__name__)

class LlamaTrainer:
    """Main trainer class for Llama 3 LoRA fine-tuning."""
    
    def __init__(self, config_path: str = "config/training_config.yaml"):
        """Initialize the trainer with configuration."""
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.device_info = None
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = self.config["logging"]["log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config["logging"]["log_level"].upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        
        logger.info("Logging setup completed")
    
    def load_model(self, model_name: str, use_4bit: bool = False, use_8bit: bool = False):
        """Load the model and tokenizer."""
        logger.info("Loading model and tokenizer...")
        
        try:
            self.model, self.tokenizer = load_model_and_tokenizer(
                model_name=model_name,
                config_path="config/model_config.yaml",
                use_4bit=use_4bit,
                use_8bit=use_8bit
            )
            
            # Get device info
            self.device_info = get_device_info()
            
            # Apply LoRA
            self.model = setup_lora(self.model, self.config_path)
            
            # Optimize for Mac Mini
            self.model = optimize_for_mac_mini(self.model, self.device_info)
            
            # Print model information
            print_model_info(self.model, self.tokenizer)
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def prepare_data(self, data_path: str) -> DatasetDict:
        """Prepare training data."""
        logger.info(f"Preparing data from: {data_path}")
        
        try:
            # Load and process data
            dataset = load_and_process_data(
                data_path=data_path,
                tokenizer=self.tokenizer,
                config_path=self.config_path
            )
            
            logger.info(f"Data prepared successfully. Train: {len(dataset['train'])} examples")
            if 'validation' in dataset:
                logger.info(f"Validation: {len(dataset['validation'])} examples")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to prepare data: {e}")
            raise
    
    def setup_training(self, dataset: DatasetDict, output_dir: str = "outputs"):
        """Setup the training configuration and trainer."""
        logger.info("Setting up training configuration...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get training arguments
        training_args = get_training_arguments(self.config_path, output_dir)
        
        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal language modeling
        )
        
        # Setup callbacks
        callbacks = []
        if self.config["training"].get("early_stopping", False):
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.01
            ))
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            data_collator=data_collator,
            callbacks=callbacks,
            tokenizer=self.tokenizer
        )
        
        logger.info("Training setup completed")
    
    def train(self) -> Dict[str, Any]:
        """Execute the training loop."""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call setup_training first.")
        
        logger.info("Starting training...")
        
        # Monitor system resources
        self._log_system_info()
        
        start_time = time.time()
        
        try:
            # Start training
            train_result = self.trainer.train()
            
            # Save the final model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.trainer.args.output_dir)
            
            # Log training results
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Training loss: {train_result.training_loss:.4f}")
            
            # Evaluate final model
            if self.trainer.eval_dataset:
                eval_results = self.trainer.evaluate()
                logger.info(f"Final evaluation results: {eval_results}")
            
            return {
                "training_loss": train_result.training_loss,
                "training_time": training_time,
                "eval_results": eval_results if self.trainer.eval_dataset else None
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _log_system_info(self):
        """Log system information for monitoring."""
        logger.info("=" * 50)
        logger.info("SYSTEM INFORMATION")
        logger.info("=" * 50)
        
        # CPU info
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        logger.info(f"CPU: {cpu_count} cores, {cpu_percent}% usage")
        
        # Memory info
        memory = psutil.virtual_memory()
        logger.info(f"Memory: {memory.total / (1024**3):.1f} GB total, "
                   f"{memory.available / (1024**3):.1f} GB available")
        
        # Disk info
        disk = psutil.disk_usage('/')
        logger.info(f"Disk: {disk.total / (1024**3):.1f} GB total, "
                   f"{disk.free / (1024**3):.1f} GB free")
        
        # Device info
        if self.device_info:
            logger.info(f"Device: {self.device_info['device']}")
            logger.info(f"Using MPS: {self.device_info['use_mps']}")
        
        logger.info("=" * 50)
    
    def save_checkpoint(self, checkpoint_dir: str):
        """Save a checkpoint of the current model."""
        if self.trainer is None:
            raise ValueError("Trainer not initialized")
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{self.trainer.state.global_step}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        self.trainer.save_model(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        logger.info(f"Checkpoint saved to: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint."""
        if self.trainer is None:
            raise ValueError("Trainer not initialized")
        
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        self.trainer = Trainer.load_from_checkpoint(checkpoint_path)
        
        # Reload model and tokenizer
        self.model = self.trainer.model
        self.tokenizer = self.trainer.tokenizer
        
        logger.info("Checkpoint loaded successfully")

# Main function removed - use train.py in project root instead
