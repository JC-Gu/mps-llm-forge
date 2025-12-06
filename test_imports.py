#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.

Run this to check if the project setup is working:
python test_imports.py
"""

import sys
from pathlib import Path

def test_imports():
    """Test all the key imports."""
    print("🧪 Testing project imports...")
    print("=" * 50)
    
    # Test 1: Basic Python packages
    print("1. Testing basic Python packages...")
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"   ❌ PyTorch: {e}")
        return False
    
    try:
        import transformers
        print(f"   ✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"   ❌ Transformers: {e}")
        return False
    
    try:
        import peft
        print(f"   ✅ PEFT: {peft.__version__}")
    except ImportError as e:
        print(f"   ❌ PEFT: {e}")
        return False
    
    # Test 2: Project modules
    print("\n2. Testing project modules...")
    
    # Add src to path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    try:
        from model_utils import get_device_info
        print("   ✅ model_utils imported successfully")
    except ImportError as e:
        print(f"   ❌ model_utils: {e}")
        return False
    
    try:
        from data_processing import DataProcessor
        print("   ✅ data_processing imported successfully")
    except ImportError as e:
        print(f"   ❌ data_processing: {e}")
        return False
    
    try:
        from training import LlamaTrainer
        print("   ✅ training imported successfully")
    except ImportError as e:
        print(f"   ❌ training: {e}")
        return False
    
    try:
        from inference import InferenceEngine
        print("   ✅ inference imported successfully")
    except ImportError as e:
        print(f"   ❌ inference: {e}")
        return False
    
    # Test 3: Hardware detection
    print("\n3. Testing hardware detection...")
    try:
        device_info = get_device_info()
        print(f"   ✅ Device: {device_info['device']}")
        print(f"   ✅ MPS available: {device_info['use_mps']}")
    except Exception as e:
        print(f"   ❌ Hardware detection failed: {e}")
        return False
    
    # Test 4: Configuration loading
    print("\n4. Testing configuration loading...")
    try:
        import yaml
        config_path = Path(__file__).parent / "config" / "training_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("   ✅ Training configuration loaded successfully")
    except Exception as e:
        print(f"   ❌ Configuration loading failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All imports and tests passed successfully!")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = test_imports()
    if not success:
        print("\n❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)
    else:
        print("\n🚀 Your project is ready to use!")
        print("\nNext steps:")
        print("1. Set your Hugging Face token: export HF_TOKEN='your_token_here'")
        print("2. Download the model: python scripts/download_model.py")
        print("3. Start training: python run_train.py")
        print("4. Run inference: python run_inference.py --model_path outputs/ --interactive")
