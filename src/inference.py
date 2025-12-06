"""
Inference module for fine-tuned Llama 3 LoRA models.

This module provides utilities for running inference with the
fine-tuned model and evaluating generation quality.
"""

import os
import logging
import json
import time
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import yaml

logger = logging.getLogger(__name__)

class InferenceEngine:
    """Engine for running inference with fine-tuned models."""
    
    def __init__(self, config_path: str = "config/training_config.yaml"):
        """Initialize the inference engine."""
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # Setup device
        self._setup_device()
        
    def _setup_device(self):
        """Setup the device for inference."""
        if torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("Using MPS device for inference")
        elif torch.cuda.is_available():
            self.device = "cuda"
            logger.info("Using CUDA device for inference")
        else:
            self.device = "cpu"
            logger.info("Using CPU device for inference")
    
    def load_model(self, model_path: str, base_model_name: Optional[str] = None):
        """
        Load the fine-tuned model and tokenizer.
        
        Args:
            model_path: Path to the fine-tuned model
            base_model_name: Base model name if loading LoRA weights separately
        """
        logger.info(f"Loading model from: {model_path}")
        
        try:
            if base_model_name:
                # Load base model first
                logger.info(f"Loading base model: {base_model_name}")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto" if self.device == "mps" else None,
                    low_cpu_mem_usage=True
                )
                
                # Load LoRA weights
                logger.info("Loading LoRA weights...")
                self.model = PeftModel.from_pretrained(base_model, model_path)
                
                # Load tokenizer from base model
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            else:
                # Load the complete fine-tuned model
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto" if self.device == "mps" else None,
                    low_cpu_mem_usage=True
                )
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Move model to device
            if self.device != "auto":
                self.model = self.model.to(self.device)
            
            # Set padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_response(
        self,
        instruction: str,
        input_text: str = "",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None
    ) -> str:
        """
        Generate a response for the given instruction and input.
        
        Args:
            instruction: The instruction to follow
            input_text: Optional input text
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty parameter
            
        Returns:
            Generated response text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model first.")
        
        # Use configuration defaults if not specified
        gen_config = self.config["generation"]
        max_new_tokens = max_new_tokens or gen_config["max_new_tokens"]
        temperature = temperature or gen_config["temperature"]
        top_p = top_p or gen_config["top_p"]
        repetition_penalty = repetition_penalty or gen_config["repetition_penalty"]
        
        # Format the prompt
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config["data"]["max_seq_length"] - max_new_tokens
        )
        
        # Move inputs to device
        if self.device != "auto":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=gen_config["do_sample"],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        generation_time = time.time() - start_time
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        logger.info(f"Generation completed in {generation_time:.2f} seconds")
        
        return response.strip()
    
    def batch_generate(
        self,
        examples: List[Dict[str, str]],
        **generation_kwargs
    ) -> List[Dict[str, str]]:
        """
        Generate responses for multiple examples.
        
        Args:
            examples: List of examples with 'instruction' and optional 'input' keys
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of examples with generated 'output' added
        """
        results = []
        
        for i, example in enumerate(examples):
            logger.info(f"Processing example {i+1}/{len(examples)}")
            
            try:
                response = self.generate_response(
                    instruction=example["instruction"],
                    input_text=example.get("input", ""),
                    **generation_kwargs
                )
                
                result = example.copy()
                result["output"] = response
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to generate response for example {i+1}: {e}")
                result = example.copy()
                result["output"] = f"ERROR: {str(e)}"
                results.append(result)
        
        return results
    
    def evaluate_perplexity(self, test_data: List[str]) -> float:
        """
        Calculate perplexity on test data.
        
        Args:
            test_data: List of test text strings
            
        Returns:
            Average perplexity
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model first.")
        
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for text in test_data:
                # Tokenize text
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config["data"]["max_seq_length"]
                )
                
                # Move to device
                if self.device != "auto":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                total_loss += loss.item() * inputs["input_ids"].shape[1]
                total_tokens += inputs["input_ids"].shape[1]
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        logger.info(f"Perplexity: {perplexity:.4f}")
        return perplexity
    
    def interactive_mode(self):
        """Run interactive inference mode."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model first.")
        
        print("\n" + "="*50)
        print("INTERACTIVE INFERENCE MODE")
        print("Type 'quit' to exit")
        print("="*50 + "\n")
        
        while True:
            try:
                # Get user input
                instruction = input("Instruction: ").strip()
                
                if instruction.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not instruction:
                    continue
                
                input_text = input("Input (optional, press Enter to skip): ").strip()
                
                # Generate response
                print("\nGenerating response...")
                response = self.generate_response(instruction, input_text)
                
                print(f"\nResponse: {response}\n")
                print("-" * 50)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error during generation: {e}")
                print(f"Error: {e}\n")
        
        print("\nExiting interactive mode.")

# Main function removed - use inference.py in project root instead
