# MPS LLM Forge: Fine-tuning & RL on Apple Silicon
 
This project provides a complete pipeline for fine-tuning and Reinforcement Learning (RL) with Large Language Models (LLMs) on Apple Silicon. The implementation is optimized for memory efficiency and performance using Metal Performance Shaders (MPS).

## Features

- **LoRA Fine-tuning**: Parameter-efficient fine-tuning with configurable rank and alpha
- **Mac Optimization**: Leverages Metal Performance Shaders (MPS) for GPU acceleration
- **Memory Efficient**: Gradient checkpointing, CPU offloading, and dynamic memory management
- **Flexible Data**: Support for various data formats and preprocessing pipelines
- **Monitoring**: Comprehensive logging with TensorBoard and WandB integration
- **Evaluation**: Built-in evaluation metrics and generation quality assessment

## Requirements

- macOS 12.3+ (for MPS support)
- Apple Silicon Mac (M1/M2/M3) with 16GB+ RAM recommended
- Python 3.8+
- 20GB+ free disk space for model and checkpoints

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mps-llm-forge
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

1. **Download the model** (first time only):
```bash
python scripts/download_model.py
```

2. **Prepare your training data** in the `data/raw/` directory

3. **Configure training parameters** in `config/training_config.yaml`

4. **Start training**:
```bash
python run_train.py
```

5. **Run inference** with your fine-tuned model:
```bash
python run_inference.py --model_path outputs/ --interactive
```

## Project Structure

```
mps-llm-forge/
├── config/          # Configuration files
├── data/            # Training data
├── src/             # Source code
├── scripts/         # Utility scripts
├── outputs/         # Training outputs
├── logs/            # Training logs
└── tests/           # Unit tests
```

## Configuration

### Training Configuration (`config/training_config.yaml`)
- Model settings (base model, LoRA parameters)
- Training hyperparameters (learning rate, batch size, etc.)
- Data processing settings
- Output and logging configuration

### LoRA Parameters
- **Rank**: 8-16 (recommended for memory efficiency)
- **Alpha**: 16-32 (scaling factor)
- **Target Modules**: q_proj, v_proj, k_proj, o_proj
- **Dropout**: 0.1

## Data Format

The training data should be in one of these formats:
- **JSON**: `{"instruction": "...", "input": "...", "output": "..."}`
- **CSV**: Columns for instruction, input, and output
- **Text**: Plain text files (one example per line)

## Training Tips

1. **Memory Management**:
   - Start with small batch sizes (1-2)
   - Use gradient accumulation (4-8 steps)
   - Enable gradient checkpointing
   - Monitor memory usage with `psutil`

2. **Performance Optimization**:
   - Use mixed precision training (FP16/BF16)
   - Leverage MPS backend for GPU acceleration
   - Optimize sequence length for your use case

3. **Monitoring**:
   - Check training logs in `logs/` directory
   - Use TensorBoard for real-time metrics
   - Monitor memory usage and training speed

## Evaluation

The project includes several evaluation metrics:
- **Perplexity**: Language model quality measure
- **Generation Quality**: Human-readable output assessment
- **Memory Usage**: Training efficiency metrics
- **Training Speed**: Tokens per second

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce batch size or sequence length
   - Enable gradient checkpointing
   - Use CPU offloading for optimizer states

2. **Slow Training**:
   - Check MPS backend availability
   - Verify mixed precision settings
   - Monitor CPU/GPU utilization

3. **Model Loading Issues**:
   - Ensure sufficient disk space
   - Check Hugging Face token permissions
   - Verify model compatibility

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Meta AI for Llama 3 models
- Hugging Face for the Transformers library
- Microsoft for PEFT implementation
- Apple for MPS backend support
