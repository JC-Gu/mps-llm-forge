#!/usr/bin/env python3
"""
Script to download Llama 3 model from Hugging Face.

This script handles the download of the base model and tokenizer
with proper authentication and error handling.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from huggingface_hub import login, snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model_utils import get_device_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_hf_token():
    """Check if Hugging Face token is available."""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    if not token:
        logger.warning("No Hugging Face token found!")
        logger.warning("Please set HF_TOKEN or HUGGING_FACE_HUB_TOKEN environment variable")
        logger.warning("You can get a token from: https://huggingface.co/settings/tokens")
        return False
    
    return True

def login_to_hf():
    """Login to Hugging Face."""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    if token:
        try:
            login(token=token)
            logger.info("Successfully logged in to Hugging Face")
            return True
        except Exception as e:
            logger.error(f"Failed to login to Hugging Face: {e}")
            return False
    
    return False

def download_model(
    model_name: str,
    output_dir: str,
    cache_dir: str = None,
    resume_download: bool = True
):
    """
    Download the model and tokenizer.
    
    Args:
        model_name: Hugging Face model name
        output_dir: Directory to save the model
        cache_dir: Cache directory for downloads
        resume_download: Whether to resume interrupted downloads
    """
    logger.info(f"Downloading model: {model_name}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            resume_download=resume_download,
            trust_remote_code=False
        )
        tokenizer.save_pretrained(output_dir)
        logger.info("Tokenizer downloaded successfully")
        
        # Download model
        logger.info("Downloading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            resume_download=resume_download,
            trust_remote_code=False,
            dtype="auto",
            low_cpu_mem_usage=True
        )
        model.save_pretrained(output_dir)
        logger.info("Model downloaded successfully")
        
        # Save model info
        model_info = {
            "model_name": model_name,
            "model_type": type(model).__name__,
            "vocab_size": model.config.vocab_size,
            "hidden_size": model.config.hidden_size,
            "num_layers": model.config.num_hidden_layers,
            "num_attention_heads": model.config.num_attention_heads,
            "max_position_embeddings": model.config.max_position_embeddings
        }
        
        info_path = os.path.join(output_dir, "model_info.json")
        import json
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Model info saved to: {info_path}")
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise

def download_with_snapshot(
    model_name: str,
    output_dir: str,
    cache_dir: str = None,
    resume_download: bool = True
):
    """
    Download using snapshot_download for better resume capability.
    
    Args:
        model_name: Hugging Face model name
        output_dir: Directory to save the model
        cache_dir: Cache directory for downloads
        resume_download: Whether to resume interrupted downloads
    """
    logger.info(f"Downloading model using snapshot: {model_name}")
    
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            cache_dir=cache_dir,
            resume_download=resume_download,
            local_dir_use_symlinks=False
        )
        logger.info("Model downloaded successfully using snapshot")
        
    except Exception as e:
        logger.error(f"Failed to download model using snapshot: {e}")
        raise

def check_disk_space(required_gb: float = 20.0):
    """Check if there's enough disk space."""
    import shutil
    
    total, used, free = shutil.disk_usage('/')
    free_gb = free / (1024**3)
    
    logger.info(f"Available disk space: {free_gb:.1f} GB")
    
    if free_gb < required_gb:
        logger.warning(f"Warning: Only {free_gb:.1f} GB available, {required_gb} GB recommended")
        return False
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download Llama 3 model")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                       help="Hugging Face model name")
    parser.add_argument("--output_dir", type=str, default="models/Meta-Llama-3-8B-Instruct",
                       help="Output directory for the model")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Cache directory for downloads")
    parser.add_argument("--use_snapshot", action="store_true",
                       help="Use snapshot_download instead of from_pretrained")
    parser.add_argument("--skip_auth", action="store_true",
                       help="Skip authentication check")
    parser.add_argument("--check_space", action="store_true",
                       help="Check disk space before downloading")
    
    args = parser.parse_args()
    
    # Check disk space
    if args.check_space:
        if not check_disk_space():
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                logger.info("Download cancelled")
                return
    
    # Check authentication
    if not args.skip_auth:
        if not check_hf_token():
            logger.error("Authentication required. Please set your Hugging Face token.")
            return
        
        if not login_to_hf():
            logger.error("Failed to authenticate with Hugging Face")
            return
    
    # Get device info
    device_info = get_device_info()
    logger.info(f"Device: {device_info['device']}")
    logger.info(f"Using MPS: {device_info['use_mps']}")
    
    try:
        if args.use_snapshot:
            download_with_snapshot(
                model_name=args.model_name,
                output_dir=args.output_dir,
                cache_dir=args.cache_dir
            )
        else:
            download_model(
                model_name=args.model_name,
                output_dir=args.output_dir,
                cache_dir=args.cache_dir
            )
        
        logger.info("=" * 50)
        logger.info("DOWNLOAD COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        logger.info(f"Model saved to: {args.output_dir}")
        logger.info("You can now use this model for fine-tuning")
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
