#!/usr/bin/env python3
"""
Quick Start Script for Llama 3 LoRA Fine-tuning Project.

This script demonstrates the basic workflow and can be used
to quickly test the project setup.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_processing import DataProcessor
from model_utils import get_device_info, print_model_info

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main quick start function."""
    print("=" * 60)
    print("LLAMA 3 LORA FINE-TUNING - QUICK START")
    print("=" * 60)
    
    try:
        # Step 1: Check system information
        print("\n1. Checking system information...")
        device_info = get_device_info()
        print(f"   Device: {device_info['device']}")
        print(f"   Using MPS: {device_info['use_mps']}")
        
        # Step 2: Check project structure
        print("\n2. Checking project structure...")
        required_dirs = ["config", "data", "src", "scripts", "outputs", "logs", "tests"]
        for dir_name in required_dirs:
            if os.path.exists(dir_name):
                print(f"   ✓ {dir_name}/")
            else:
                print(f"   ✗ {dir_name}/ (missing)")
        
        # Step 3: Check configuration files
        print("\n3. Checking configuration files...")
        config_files = ["config/training_config.yaml", "config/model_config.yaml"]
        for config_file in config_files:
            if os.path.exists(config_file):
                print(f"   ✓ {config_file}")
            else:
                print(f"   ✗ {config_file} (missing)")
        
        # Step 4: Check sample data
        print("\n4. Checking sample data...")
        sample_data_file = "data/raw/sample_data.json"
        if os.path.exists(sample_data_file):
            print(f"   ✓ {sample_data_file}")
            # Try to load the data
            try:
                processor = DataProcessor()
                sample_data = processor.create_sample_data()
                print(f"   ✓ Sample data created with {len(sample_data)} examples")
            except Exception as e:
                print(f"   ✗ Failed to create sample data: {e}")
        else:
            print(f"   ✗ {sample_data_file} (missing)")
        
        # Step 5: Check source code
        print("\n5. Checking source code...")
        source_files = [
            "src/__init__.py",
            "src/model_utils.py", 
            "src/data_processing.py",
            "src/training.py",
            "src/inference.py"
        ]
        for source_file in source_files:
            if os.path.exists(source_file):
                print(f"   ✓ {source_file}")
            else:
                print(f"   ✗ {source_file} (missing)")
        
        # Step 6: Check scripts
        print("\n6. Checking utility scripts...")
        script_files = [
            "scripts/download_model.py",
            "scripts/evaluate.py"
        ]
        for script_file in script_files:
            if os.path.exists(script_file):
                print(f"   ✓ {script_file}")
            else:
                print(f"   ✗ {script_file} (missing)")
        
        # Step 7: Summary and next steps
        print("\n" + "=" * 60)
        print("QUICK START COMPLETE!")
        print("=" * 60)
        
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set your Hugging Face token: export HF_TOKEN='your_token_here'")
        print("3. Download the model: python scripts/download_model.py")
        print("4. Start training: python run_train.py")
        print("5. Run inference: python run_inference.py --model_path outputs/")
        
        print("\nFor more information, see README.md")
        
        # Check if requirements.txt exists
        if os.path.exists("requirements.txt"):
            print("\n✓ requirements.txt found - ready for dependency installation")
        else:
            print("\n✗ requirements.txt missing - please create it first")
        
    except Exception as e:
        logger.error(f"Quick start failed: {e}")
        print(f"\n✗ Quick start failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
