#!/usr/bin/env python3
"""
Direct inference script for fine-tuned Llama 3 LoRA models.

This script can be run directly from the project root:
python inference.py --model_path outputs/ --interactive
"""

import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Now import the inference module
from inference import InferenceEngine

def main():
    """Main inference function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned Llama 3")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the fine-tuned model")
    parser.add_argument("--base_model", type=str, default=None,
                       help="Base model name (if loading LoRA weights separately)")
    parser.add_argument("--instruction", type=str, default="",
                       help="Instruction for generation")
    parser.add_argument("--input_text", type=str, default="",
                       help="Input text for generation")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--config_path", type=str, default="config/training_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=None,
                       help="Sampling temperature")
    
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = InferenceEngine(args.config_path)
    
    try:
        # Load model
        engine.load_model(args.model_path, args.base_model)
        
        if args.interactive:
            # Run interactive mode
            engine.interactive_mode()
        else:
            # Generate single response
            if not args.instruction:
                print("Please provide an instruction or use --interactive mode")
                return
            
            response = engine.generate_response(
                instruction=args.instruction,
                input_text=args.input_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature
            )
            
            print(f"\nInstruction: {args.instruction}")
            if args.input_text:
                print(f"Input: {args.input_text}")
            print(f"Response: {response}")
            
    except Exception as e:
        print(f"Inference failed: {e}")
        raise

if __name__ == "__main__":
    main()
