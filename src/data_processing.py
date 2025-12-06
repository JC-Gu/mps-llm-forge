"""
Data processing utilities for Llama 3 LoRA fine-tuning.

This module handles data loading, preprocessing, and tokenization
for various data formats commonly used in instruction fine-tuning.
"""

import os
import json
import csv
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer
import yaml

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data loading and preprocessing for training."""
    
    def __init__(self, config_path: str = "config/training_config.yaml"):
        """Initialize the data processor with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config["data"]
        self.max_seq_length = self.data_config["max_seq_length"]
        
    def load_data(self, data_path: str) -> DatasetDict:
        """
        Load data from various formats and return a DatasetDict.
        
        Args:
            data_path: Path to the data file or directory
            
        Returns:
            DatasetDict with train and validation splits
        """
        logger.info(f"Loading data from: {data_path}")
        
        if os.path.isdir(data_path):
            return self._load_from_directory(data_path)
        else:
            return self._load_from_file(data_path)
    
    def _load_from_file(self, file_path: str) -> DatasetDict:
        """Load data from a single file."""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.json':
            return self._load_json(file_path)
        elif file_ext == '.csv':
            return self._load_csv(file_path)
        elif file_ext == '.txt':
            return self._load_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _load_from_directory(self, dir_path: str) -> DatasetDict:
        """Load data from a directory containing multiple files."""
        data_files = []
        
        for file_path in Path(dir_path).glob("*"):
            if file_path.suffix.lower() in ['.json', '.csv', '.txt']:
                data_files.append(str(file_path))
        
        if not data_files:
            raise ValueError(f"No supported data files found in {dir_path}")
        
        # Load all files and combine
        datasets = []
        for file_path in data_files:
            try:
                dataset = self._load_from_file(file_path)
                datasets.append(dataset)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue
        
        if not datasets:
            raise ValueError("No datasets could be loaded")
        
        # Combine datasets
        return self._combine_datasets(datasets)
    
    def _load_json(self, file_path: str) -> DatasetDict:
        """Load data from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # List of examples
            examples = data
        elif isinstance(data, dict) and "examples" in data:
            # Dict with examples key
            examples = data["examples"]
        else:
            raise ValueError("Invalid JSON format. Expected list or dict with 'examples' key")
        
        # Convert to dataset
        dataset = Dataset.from_list(examples)
        return self._create_train_val_split(dataset)
    
    def _load_csv(self, file_path: str) -> DatasetDict:
        """Load data from CSV file."""
        df = pd.read_csv(file_path)
        
        # Check required columns
        required_cols = [self.data_config["instruction_column"]]
        if self.data_config["input_column"] in df.columns:
            required_cols.append(self.data_config["input_column"])
        required_cols.append(self.data_config["output_column"])
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert to dataset
        dataset = Dataset.from_pandas(df)
        return self._create_train_val_split(dataset)
    
    def _load_text(self, file_path: str) -> DatasetDict:
        """Load data from text file (one example per line)."""
        examples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        # Try to parse as JSON first
                        example = json.loads(line)
                        examples.append(example)
                    except json.JSONDecodeError:
                        # Treat as plain text
                        examples.append({
                            "text": line,
                            "instruction": "Continue the text",
                            "input": "",
                            "output": line
                        })
        
        if not examples:
            raise ValueError(f"No valid examples found in {file_path}")
        
        dataset = Dataset.from_list(examples)
        return self._create_train_val_split(dataset)
    
    def _create_train_val_split(self, dataset: Dataset) -> DatasetDict:
        """Create train/validation split from dataset."""
        split_percentage = self.data_config["validation_split_percentage"]
        
        if split_percentage > 0:
            dataset = dataset.train_test_split(
                test_size=split_percentage / 100,
                seed=42
            )
            return DatasetDict({
                "train": dataset["train"],
                "validation": dataset["test"]
            })
        else:
            return DatasetDict({"train": dataset})
    
    def _combine_datasets(self, datasets: List[DatasetDict]) -> DatasetDict:
        """Combine multiple datasets."""
        combined_train = []
        combined_val = []
        
        for dataset in datasets:
            if "train" in dataset:
                combined_train.append(dataset["train"])
            if "validation" in dataset:
                combined_val.append(dataset["validation"])
        
        # Combine train datasets
        if combined_train:
            train_dataset = Dataset.concatenate_datasets(combined_train)
        else:
            raise ValueError("No training data found")
        
        # Combine validation datasets
        if combined_val:
            val_dataset = Dataset.concatenate_datasets(combined_val)
        else:
            val_dataset = None
        
        result = DatasetDict({"train": train_dataset})
        if val_dataset:
            result["validation"] = val_dataset
        
        return result
    
    def preprocess_data(self, dataset: DatasetDict, tokenizer: PreTrainedTokenizer) -> DatasetDict:
        """
        Preprocess the dataset for training.
        
        Args:
            dataset: Input dataset
            tokenizer: Tokenizer for text processing
            
        Returns:
            Preprocessed dataset
        """
        logger.info("Preprocessing dataset...")
        
        def format_instruction(example):
            """Format example into instruction format."""
            instruction = example.get(self.data_config["instruction_column"], "")
            input_text = example.get(self.data_config["input_column"], "")
            output = example.get(self.data_config["output_column"], "")
            
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            
            return {"text": prompt}
        
        def tokenize_function(examples):
            """Tokenize the examples."""
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.max_seq_length,
                return_tensors=None
            )
        
        # Apply formatting
        processed_dataset = {}
        for split_name, split_dataset in dataset.items():
            logger.info(f"Processing {split_name} split...")
            
            # Format examples
            formatted_dataset = split_dataset.map(
                format_instruction,
                remove_columns=split_dataset.column_names,
                desc=f"Formatting {split_name}"
            )
            
            # Tokenize
            tokenized_dataset = formatted_dataset.map(
                tokenize_function,
                remove_columns=formatted_dataset.column_names,
                desc=f"Tokenizing {split_name}"
            )
            
            processed_dataset[split_name] = tokenized_dataset
        
        logger.info("Dataset preprocessing completed")
        return DatasetDict(processed_dataset)
    
    def create_sample_data(self, output_dir: str = "data/raw"):
        """Create sample training data for testing."""
        os.makedirs(output_dir, exist_ok=True)
        
        sample_data = [
            {
                "instruction": "Explain what machine learning is",
                "input": "",
                "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on those patterns."
            },
            {
                "instruction": "Write a Python function to calculate factorial",
                "input": "",
                "output": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n-1)"
            },
            {
                "instruction": "Translate the following text to Spanish",
                "input": "Hello, how are you today?",
                "output": "Hola, ¿cómo estás hoy?"
            },
            {
                "instruction": "Summarize the main benefits of renewable energy",
                "input": "",
                "output": "Renewable energy offers several key benefits: it's sustainable and won't run out, produces minimal greenhouse gas emissions, creates jobs in the energy sector, reduces dependence on fossil fuels, and can provide energy security for countries."
            },
            {
                "instruction": "Solve the math problem",
                "input": "What is 15% of 200?",
                "output": "15% of 200 is 30. To calculate this: 200 × 0.15 = 30"
            }
        ]
        
        # Save as JSON
        json_path = os.path.join(output_dir, "sample_data.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
        # Save as CSV
        csv_path = os.path.join(output_dir, "sample_data.csv")
        df = pd.DataFrame(sample_data)
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Sample data created in {output_dir}")
        logger.info(f"Files created: {json_path}, {csv_path}")
        
        return sample_data

def load_and_process_data(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    config_path: str = "config/training_config.yaml"
) -> DatasetDict:
    """
    Convenience function to load and process data in one step.
    
    Args:
        data_path: Path to the data
        tokenizer: Tokenizer for processing
        config_path: Path to configuration file
        
    Returns:
        Processed DatasetDict ready for training
    """
    processor = DataProcessor(config_path)
    dataset = processor.load_data(data_path)
    return processor.preprocess_data(dataset, tokenizer)

def validate_data_format(data_path: str) -> bool:
    """
    Validate that the data format is correct.
    
    Args:
        data_path: Path to the data file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        processor = DataProcessor()
        processor.load_data(data_path)
        return True
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return False
