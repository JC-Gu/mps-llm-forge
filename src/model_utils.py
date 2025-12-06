"""
Model utilities for Llama 3 LoRA fine-tuning on Mac Mini.

This module provides functions for loading models, setting up LoRA,
and optimizing for Apple Silicon hardware.
"""

import os
import torch
import logging
from typing import Dict, Any, Optional, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import yaml

logger = logging.getLogger(__name__)

def get_device_info() -> Dict[str, Any]:
    """Get device information and capabilities for Mac Mini."""
    device_info = {
        "device": "cpu",
        "use_mps": False,
        "use_cpu_offload": False,
        "max_memory": "14GB"
    }
    
    # Check MPS availability
    if torch.backends.mps.is_available():
        device_info["device"] = "mps"
        device_info["use_mps"] = True
        logger.info("MPS (Metal Performance Shaders) is available")
    else:
        logger.warning("MPS not available, falling back to CPU")
    
    # Check CUDA availability (unlikely on Mac Mini, but good to check)
    if torch.cuda.is_available():
        device_info["device"] = "cuda"
        device_info["use_mps"] = False
        logger.info("CUDA is available")
    
    return device_info

def load_model_and_tokenizer(
    model_name: str,
    config_path: str = "config/model_config.yaml",
    use_4bit: bool = False,
    use_8bit: bool = False
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load Llama 3 model and tokenizer with Mac Mini optimizations.
    
    Args:
        model_name: Hugging Face model name
        config_path: Path to model configuration file
        use_4bit: Whether to use 4-bit quantization
        use_8bit: Whether to use 8-bit quantization
    
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get device info
    device_info = get_device_info()
    
    # Configure quantization if requested
    quantization_config = None
    if use_4bit or use_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            load_in_8bit=use_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config.get("trust_remote_code", False),
        model_max_length=config["tokenizer"]["model_max_length"],
        padding_side=config["tokenizer"]["padding_side"],
        truncation_side=config["tokenizer"]["truncation_side"],
        use_fast=config["tokenizer"]["use_fast"]
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Added EOS token as padding token")
    
    # Load model
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="auto" if device_info["use_mps"] else None,
        low_cpu_mem_usage=True,
        trust_remote_code=config.get("trust_remote_code", False)
    )
    
    # Prepare model for training
    if use_4bit or use_8bit:
        model = prepare_model_for_kbit_training(model)
        logger.info("Model prepared for k-bit training")
    
    # Enable gradient checkpointing for memory efficiency
    if config["training"]["use_gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Disable cache during training
    model.config.use_cache = False
    
    logger.info(f"Model loaded successfully on device: {device_info['device']}")
    return model, tokenizer

def setup_lora(
    model: AutoModelForCausalLM,
    config_path: str = "config/training_config.yaml"
) -> AutoModelForCausalLM:
    """
    Set up LoRA configuration for the model.
    
    Args:
        model: The base model to apply LoRA to
        config_path: Path to training configuration file
    
    Returns:
        Model with LoRA applied
    """
    logger.info("Setting up LoRA configuration...")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    lora_config = config["lora"]
    
    # Create LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        target_modules=lora_config["target_modules"],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config["bias"]
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    logger.info("LoRA configuration applied successfully")
    return model

def optimize_for_mac_mini(
    model: AutoModelForCausalLM,
    device_info: Dict[str, Any]
) -> AutoModelForCausalLM:
    """
    Apply Mac Mini specific optimizations to the model.
    
    Args:
        model: The model to optimize
        device_info: Device information dictionary
    
    Returns:
        Optimized model
    """
    logger.info("Applying Mac Mini optimizations...")
    
    if device_info["use_mps"]:
        # Move model to MPS device
        model = model.to("mps")
        logger.info("Model moved to MPS device")
        
        # Enable memory efficient attention if available
        try:
            from transformers.models.llama.modeling_llama import LlamaAttention
            for module in model.modules():
                if isinstance(module, LlamaAttention):
                    module._attn_implementation = "flash_attention_2"
            logger.info("Flash attention 2 enabled")
        except ImportError:
            logger.info("Flash attention 2 not available, using standard attention")
    
    return model

def get_training_arguments(
    config_path: str = "config/training_config.yaml",
    output_dir: str = "outputs"
) -> TrainingArguments:
    """
    Create training arguments from configuration.
    
    Args:
        config_path: Path to training configuration file
        output_dir: Output directory for training artifacts
    
    Returns:
        TrainingArguments object
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    training_config = config["training"]
    
    # Ensure numeric types are correct
    learning_rate = float(training_config["learning_rate"])
    weight_decay = float(training_config["weight_decay"])
    max_grad_norm = float(training_config["max_grad_norm"])
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(training_config["num_train_epochs"]),
        per_device_train_batch_size=int(training_config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training_config["gradient_accumulation_steps"]),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=int(training_config["warmup_steps"]),
        max_steps=int(training_config["max_steps"]),
        save_steps=int(training_config["save_steps"]),
        eval_steps=int(training_config["eval_steps"]),
        logging_steps=int(training_config["logging_steps"]),
        lr_scheduler_type=str(training_config["lr_scheduler_type"]),
        max_grad_norm=max_grad_norm,
        gradient_checkpointing=bool(training_config["gradient_checkpointing"]),
        dataloader_pin_memory=bool(training_config["dataloader_pin_memory"]),
        remove_unused_columns=bool(training_config["remove_unused_columns"]),
        fp16=bool(training_config["fp16"]),
        bf16=bool(training_config["bf16"]),
        save_strategy=str(config["output"]["save_strategy"]),
        eval_strategy=str(config["output"]["eval_strategy"]),
        save_total_limit=int(config["output"]["save_total_limit"]),
        load_best_model_at_end=bool(config["output"]["load_best_model_at_end"]),
        metric_for_best_model=str(config["output"]["metric_for_best_model"]),
        greater_is_better=bool(config["output"]["greater_is_better"]),
        save_safetensors=bool(config["output"]["save_safetensors"]),
        logging_dir=str(config["logging"]["logging_dir"]),
        logging_first_step=bool(config["logging"]["logging_first_step"]),
        report_to=["tensorboard"] if config["logging"]["tensorboard_dir"] else [],
        run_name="llama3-lora-training"
    )

def print_model_info(model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    """Print model and tokenizer information."""
    logger.info("=" * 50)
    logger.info("MODEL INFORMATION")
    logger.info("=" * 50)
    logger.info(f"Model type: {type(model).__name__}")
    logger.info(f"Tokenizer type: {type(tokenizer).__name__}")
    logger.info(f"Vocabulary size: {tokenizer.vocab_size}")
    logger.info(f"Model max length: {tokenizer.model_max_length}")
    logger.info(f"Padding token: {tokenizer.pad_token}")
    logger.info(f"EOS token: {tokenizer.eos_token}")
    logger.info(f"BOS token: {tokenizer.bos_token}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Parameter efficiency: {trainable_params/total_params*100:.2f}%")
    logger.info("=" * 50)
