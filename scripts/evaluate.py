#!/usr/bin/env python3
"""
Evaluation script for fine-tuned Llama 3 LoRA models.

This script provides comprehensive evaluation of the fine-tuned model
including perplexity, generation quality, and performance metrics.
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any
import torch
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from inference import InferenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluator."""
    
    def __init__(self, config_path: str = "config/training_config.yaml"):
        """Initialize the evaluator."""
        self.config_path = config_path
        self.inference_engine = InferenceEngine(config_path)
        
    def load_model(self, model_path: str, base_model_name: str = None):
        """Load the model for evaluation."""
        self.inference_engine.load_model(model_path, base_model_name)
    
    def evaluate_perplexity(self, test_data: List[str]) -> Dict[str, float]:
        """Evaluate model perplexity on test data."""
        logger.info("Evaluating perplexity...")
        
        start_time = time.time()
        perplexity = self.inference_engine.evaluate_perplexity(test_data)
        eval_time = time.time() - start_time
        
        results = {
            "perplexity": perplexity,
            "eval_time": eval_time,
            "num_samples": len(test_data)
        }
        
        logger.info(f"Perplexity evaluation completed in {eval_time:.2f} seconds")
        return results
    
    def evaluate_generation_quality(
        self,
        test_cases: List[Dict[str, str]],
        **generation_kwargs
    ) -> Dict[str, Any]:
        """Evaluate generation quality on test cases."""
        logger.info("Evaluating generation quality...")
        
        results = []
        total_time = 0
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"Evaluating test case {i+1}/{len(test_cases)}")
            
            start_time = time.time()
            try:
                response = self.inference_engine.generate_response(
                    instruction=test_case["instruction"],
                    input_text=test_case.get("input", ""),
                    **generation_kwargs
                )
                
                generation_time = time.time() - start_time
                total_time += generation_time
                
                result = {
                    "test_case": test_case,
                    "generated_response": response,
                    "generation_time": generation_time,
                    "success": True
                }
                
            except Exception as e:
                generation_time = time.time() - start_time
                total_time += generation_time
                
                result = {
                    "test_case": test_case,
                    "generated_response": f"ERROR: {str(e)}",
                    "generation_time": generation_time,
                    "success": False,
                    "error": str(e)
                }
            
            results.append(result)
        
        # Calculate metrics
        successful_generations = [r for r in results if r["success"]]
        avg_generation_time = total_time / len(test_cases) if test_cases else 0
        
        evaluation_results = {
            "results": results,
            "total_samples": len(test_cases),
            "successful_generations": len(successful_generations),
            "success_rate": len(successful_generations) / len(test_cases) if test_cases else 0,
            "total_time": total_time,
            "avg_generation_time": avg_generation_time,
            "tokens_per_second": self._calculate_tokens_per_second(results)
        }
        
        logger.info(f"Generation quality evaluation completed")
        logger.info(f"Success rate: {evaluation_results['success_rate']:.2%}")
        logger.info(f"Average generation time: {avg_generation_time:.2f} seconds")
        
        return evaluation_results
    
    def _calculate_tokens_per_second(self, results: List[Dict[str, Any]]) -> float:
        """Calculate average tokens per second."""
        if not results:
            return 0.0
        
        total_tokens = 0
        total_time = 0
        
        for result in results:
            if result["success"]:
                # Estimate tokens (rough approximation)
                response = result["generated_response"]
                estimated_tokens = len(response.split()) * 1.3  # Rough token-to-word ratio
                total_tokens += estimated_tokens
                total_time += result["generation_time"]
        
        if total_time > 0:
            return total_tokens / total_time
        return 0.0
    
    def run_comprehensive_evaluation(
        self,
        test_data: List[str],
        test_cases: List[Dict[str, str]],
        output_dir: str = "evaluation_results"
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation."""
        logger.info("Starting comprehensive evaluation...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Run evaluations
        perplexity_results = self.evaluate_perplexity(test_data)
        generation_results = self.evaluate_generation_quality(test_cases)
        
        # Combine results
        comprehensive_results = {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "perplexity_evaluation": perplexity_results,
            "generation_evaluation": generation_results,
            "summary": {
                "perplexity": perplexity_results["perplexity"],
                "success_rate": generation_results["success_rate"],
                "avg_generation_time": generation_results["avg_generation_time"],
                "tokens_per_second": generation_results["tokens_per_second"]
            }
        }
        
        # Save results
        results_file = os.path.join(output_dir, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to: {results_file}")
        
        # Generate plots
        self._generate_evaluation_plots(comprehensive_results, output_dir)
        
        return comprehensive_results
    
    def _generate_evaluation_plots(self, results: Dict[str, Any], output_dir: str):
        """Generate evaluation plots."""
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Model Evaluation Results', fontsize=16)
            
            # Plot 1: Perplexity
            axes[0, 0].bar(['Perplexity'], [results['summary']['perplexity']], 
                           color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Model Perplexity')
            axes[0, 0].set_ylabel('Perplexity')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Success Rate
            axes[0, 1].pie([results['summary']['success_rate'], 
                           1 - results['summary']['success_rate']], 
                           labels=['Success', 'Failure'], 
                           autopct='%1.1f%%', 
                           colors=['lightgreen', 'lightcoral'])
            axes[0, 1].set_title('Generation Success Rate')
            
            # Plot 3: Generation Time Distribution
            generation_times = [r['generation_time'] for r in results['generation_evaluation']['results']]
            axes[1, 0].hist(generation_times, bins=20, color='lightblue', alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Generation Time Distribution')
            axes[1, 0].set_xlabel('Time (seconds)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Performance Metrics
            metrics = ['Success Rate', 'Tokens/Second', 'Avg Gen Time']
            values = [
                results['summary']['success_rate'],
                results['summary']['tokens_per_second'],
                1 / results['summary']['avg_generation_time'] if results['summary']['avg_generation_time'] > 0 else 0
            ]
            
            # Normalize values for better visualization
            normalized_values = [v / max(values) if max(values) > 0 else 0 for v in values]
            
            axes[1, 1].bar(metrics, normalized_values, color=['lightgreen', 'lightblue', 'lightcoral'], alpha=0.7)
            axes[1, 1].set_title('Performance Metrics (Normalized)')
            axes[1, 1].set_ylabel('Normalized Value')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(output_dir, "evaluation_plots.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation plots saved to: {plot_file}")
            
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """Print a summary of evaluation results."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Timestamp: {results['evaluation_timestamp']}")
        print(f"Perplexity: {results['summary']['perplexity']:.4f}")
        print(f"Success Rate: {results['summary']['success_rate']:.2%}")
        print(f"Average Generation Time: {results['summary']['avg_generation_time']:.3f} seconds")
        print(f"Tokens per Second: {results['summary']['tokens_per_second']:.2f}")
        print("="*60)

def create_sample_test_cases() -> List[Dict[str, str]]:
    """Create sample test cases for evaluation."""
    return [
        {
            "instruction": "Explain what artificial intelligence is",
            "input": "",
            "expected_keywords": ["machine learning", "neural networks", "automation"]
        },
        {
            "instruction": "Write a Python function to calculate fibonacci numbers",
            "input": "",
            "expected_keywords": ["def", "fibonacci", "return"]
        },
        {
            "instruction": "Translate the following to French",
            "input": "Hello, how are you?",
            "expected_keywords": ["bonjour", "comment", "allez-vous"]
        },
        {
            "instruction": "Summarize the benefits of renewable energy",
            "input": "",
            "expected_keywords": ["sustainable", "clean", "environment"]
        },
        {
            "instruction": "Solve the math problem step by step",
            "input": "What is 25% of 80?",
            "expected_keywords": ["25%", "80", "20", "calculation"]
        }
    ]

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Llama 3 model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the fine-tuned model")
    parser.add_argument("--base_model", type=str, default=None,
                       help="Base model name (if loading LoRA weights separately)")
    parser.add_argument("--test_data", type=str, default=None,
                       help="Path to test data file")
    parser.add_argument("--test_cases", type=str, default=None,
                       help="Path to test cases file")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Output directory for evaluation results")
    parser.add_argument("--config_path", type=str, default="config/training_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                       help="Maximum new tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.config_path)
    
    try:
        # Load model
        logger.info("Loading model for evaluation...")
        evaluator.load_model(args.model_path, args.base_model)
        
        # Prepare test data
        test_data = []
        if args.test_data and os.path.exists(args.test_data):
            with open(args.test_data, 'r') as f:
                test_data = [line.strip() for line in f if line.strip()]
        else:
            # Use sample data if no test data provided
            test_data = [
                "Artificial intelligence is transforming the world.",
                "Machine learning algorithms can identify patterns in data.",
                "Neural networks are inspired by biological brain structures."
            ]
        
        # Prepare test cases
        test_cases = []
        if args.test_cases and os.path.exists(args.test_cases):
            with open(args.test_cases, 'r') as f:
                test_cases = json.load(f)
        else:
            # Use sample test cases
            test_cases = create_sample_test_cases()
        
        logger.info(f"Using {len(test_data)} test data samples")
        logger.info(f"Using {len(test_cases)} test cases")
        
        # Run evaluation
        results = evaluator.run_comprehensive_evaluation(
            test_data=test_data,
            test_cases=test_cases,
            output_dir=args.output_dir
        )
        
        # Print summary
        evaluator.print_evaluation_summary(results)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
