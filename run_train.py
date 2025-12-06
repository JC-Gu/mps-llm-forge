#!/usr/bin/env python3
"""
Direct training script for Llama 3 LoRA fine-tuning.

This script can be run directly from the project root:
python train.py
"""

import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Now import the training module
from training import LlamaTrainer

def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Llama 3 with LoRA")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                       help="Hugging Face model name")
    parser.add_argument("--data_path", type=str, default="data/raw/sample_data.json",
                       help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory for training artifacts")
    parser.add_argument("--use_4bit", action="store_true",
                       help="Use 4-bit quantization")
    parser.add_argument("--use_8bit", action="store_true",
                       help="Use 8-bit quantization")
    parser.add_argument("--config_path", type=str, default="config/training_config.yaml",
                       help="Path to training configuration")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = LlamaTrainer(args.config_path)
    
    try:
        # Load model
        trainer.load_model(args.model_name, args.use_4bit, args.use_8bit)
        
        # Prepare data
        dataset = trainer.prepare_data(args.data_path)
        
        # Setup training
        trainer.setup_training(dataset, args.output_dir)
        
        # Start training
        results = trainer.train()
        
        print("Training completed successfully!")
        print(f"Results: {results}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
