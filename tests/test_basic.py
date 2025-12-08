"""
Basic tests for the Llama 3 LoRA fine-tuning project.

These tests verify that the project structure is correct and
basic imports work without errors.
"""

import unittest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

class TestProjectStructure(unittest.TestCase):
    """Test that the project structure is correct."""
    
    def test_config_files_exist(self):
        """Test that configuration files exist."""
        config_dir = Path(__file__).parent.parent / "config"
        
        self.assertTrue((config_dir / "training_config.yaml").exists())
        self.assertTrue((config_dir / "model_config.yaml").exists())
    
    def test_source_files_exist(self):
        """Test that source files exist."""
        src_dir = Path(__file__).parent.parent / "src"
        
        self.assertTrue((src_dir / "__init__.py").exists())
        self.assertTrue((src_dir / "model_utils.py").exists())
        self.assertTrue((src_dir / "data_processing.py").exists())
        self.assertTrue((src_dir / "training.py").exists())
        self.assertTrue((src_dir / "inference.py").exists())
    
    def test_script_files_exist(self):
        """Test that script files exist."""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        
        self.assertTrue((scripts_dir / "download_model.py").exists())
        self.assertTrue((scripts_dir / "evaluate.py").exists())
    
    def test_data_files_exist(self):
        """Test that sample data files exist."""
        data_dir = Path(__file__).parent.parent / "data" / "raw"
        
        self.assertTrue((data_dir / "sample_data.json").exists())

class TestImports(unittest.TestCase):
    """Test that basic imports work."""
    
    def test_basic_imports(self):
        """Test that basic Python packages can be imported."""
        try:
            import torch
            import transformers
            import peft
            import datasets
            import yaml
            import pandas
            import numpy
        except ImportError as e:
            self.fail(f"Failed to import required package: {e}")
    
    def test_project_imports(self):
        """Test that project modules can be imported."""
        try:
            from src import model_utils
            from src import data_processing
            from src import training
            from src import inference
        except ImportError as e:
            self.fail(f"Failed to import project module: {e}")

class TestConfiguration(unittest.TestCase):
    """Test configuration file loading."""
    
    def test_training_config_loading(self):
        """Test that training configuration can be loaded."""
        try:
            import yaml
            config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            self.assertIn("model", config)
            self.assertIn("lora", config)
            self.assertIn("training", config)
            self.assertIn("data", config)
            
        except Exception as e:
            self.fail(f"Failed to load training configuration: {e}")
    
    def test_model_config_loading(self):
        """Test that model configuration can be loaded."""
        try:
            import yaml
            config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            self.assertIn("architecture", config)
            self.assertIn("tokenizer", config)
            self.assertIn("training", config)
            
        except Exception as e:
            self.fail(f"Failed to load model configuration: {e}")

class TestSampleData(unittest.TestCase):
    """Test sample data loading."""
    
    def test_sample_data_loading(self):
        """Test that sample data can be loaded."""
        try:
            import json
            data_path = Path(__file__).parent.parent / "data" / "raw" / "sample_data.json"
            
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            # Check data structure
            self.assertIsInstance(data, list)
            self.assertGreater(len(data), 0)
            
            # Check first example structure
            first_example = data[0]
            self.assertIn("instruction", first_example)
            self.assertIn("input", first_example)
            self.assertIn("output", first_example)
            
        except Exception as e:
            self.fail(f"Failed to load sample data: {e}")

if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTest(unittest.makeSuite(TestProjectStructure))
    test_suite.addTest(unittest.makeSuite(TestImports))
    test_suite.addTest(unittest.makeSuite(TestConfiguration))
    test_suite.addTest(unittest.makeSuite(TestSampleData))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(not result.wasSuccessful())
