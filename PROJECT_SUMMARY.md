# Project Summary: MPS LLM Forge (Fine-tuning & RL on Apple Silicon)
 
 ## 🎯 Project Overview
 
 This project provides a complete, production-ready pipeline for fine-tuning and Reinforcement Learning (RL) with Large Language Models (LLMs) on Apple Silicon. The implementation is specifically optimized for memory efficiency and performance on Apple hardware using Metal Performance Shaders (MPS).

## 🏗️ Project Structure

```
mps-llm-forge/
├── 📁 config/                    # Configuration files
│   ├── training_config.yaml      # Main training configuration
│   └── model_config.yaml         # Model-specific configuration
├── 📁 data/                      # Data directory
│   ├── raw/                      # Raw training data
│   │   └── sample_data.json      # Sample training examples
│   └── processed/                # Preprocessed data (auto-generated)
├── 📁 src/                       # Source code
│   ├── __init__.py               # Package initialization
│   ├── model_utils.py            # Model loading and LoRA setup
│   ├── data_processing.py        # Data preprocessing utilities
│   ├── training.py               # Main training module
│   └── inference.py              # Inference and evaluation
├── 📁 scripts/                   # Utility scripts
│   ├── download_model.py         # Model download script
│   └── evaluate.py               # Model evaluation script
├── 📁 outputs/                   # Training outputs and checkpoints
├── 📁 logs/                      # Training logs
├── 📁 tests/                     # Unit tests
├── requirements.txt               # Python dependencies
├── setup.py                      # Installation script
├── quick_start.py                # Quick start demonstration
└── README.md                     # Comprehensive documentation
```

## 🚀 Key Features

### 1. **LoRA Fine-tuning**
- **Parameter Efficiency**: Only trains 16-32 parameters per original parameter
- **Configurable Rank**: Adjustable LoRA rank (8-16 recommended for Mac Mini)
- **Target Modules**: Optimized for attention layers (q_proj, v_proj, k_proj, o_proj)
- **Memory Optimization**: No bias training, configurable dropout

### 2. **Mac Mini Optimization**
- **MPS Backend**: Leverages Metal Performance Shaders for GPU acceleration
- **Memory Management**: Gradient checkpointing, CPU offloading, dynamic allocation
- **Mixed Precision**: BF16 training for better numerical stability
- **Hardware Detection**: Automatic device detection and optimization

### 3. **Data Processing**
- **Multiple Formats**: Support for JSON, CSV, and text files
- **Flexible Schema**: Configurable instruction/input/output columns
- **Automatic Splitting**: Train/validation split with configurable ratio
- **Tokenization**: Optimized for Llama 3 tokenizer

### 4. **Training Pipeline**
- **Configurable Parameters**: All training hyperparameters in YAML files
- **Monitoring**: Comprehensive logging with TensorBoard support
- **Checkpointing**: Automatic model saving and resumption
- **Early Stopping**: Configurable early stopping with patience

### 5. **Inference & Evaluation**
- **Interactive Mode**: Command-line interactive inference
- **Batch Processing**: Process multiple examples efficiently
- **Quality Metrics**: Perplexity calculation and generation evaluation
- **Performance Profiling**: Generation speed and memory usage tracking

## ⚙️ Technical Specifications

### **Model Configuration**
- **Base Model**: Llama 3 8B (meta-llama/Meta-Llama-3-8B-Instruct)
- **Architecture**: 32 layers, 4096 hidden size, 32 attention heads
- **Vocabulary**: 128,256 tokens
- **Context Length**: Up to 8,192 tokens (configurable)

### **LoRA Configuration**
- **Rank**: 16 (configurable)
- **Alpha**: 32 (scaling factor)
- **Target Modules**: q_proj, v_proj, k_proj, o_proj
- **Dropout**: 0.1
- **Bias**: None (memory optimization)

### **Training Configuration**
- **Batch Size**: 1-2 (memory-optimized for Mac Mini)
- **Gradient Accumulation**: 8 steps (effective batch size: 8-16)
- **Learning Rate**: 2e-4 (configurable)
- **Warmup Steps**: 100
- **Max Steps**: 5,000
- **Sequence Length**: 2,048 (configurable)

### **Hardware Requirements**
- **Minimum**: macOS 12.3+, 16GB RAM, 20GB free disk space
- **Recommended**: Apple Silicon Mac (M1/M2/M3), 32GB+ RAM
- **Storage**: 20GB+ for model, 10GB+ for checkpoints

## 🛠️ Installation & Setup

### 1. **Environment Setup**
```bash
# Clone the repository
git clone <repository-url>
cd mps-llm-forge

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Hugging Face Authentication**
```bash
# Set your Hugging Face token
export HF_TOKEN="your_token_here"

# Or set it permanently in your shell profile
echo 'export HF_TOKEN="your_token_here"' >> ~/.zshrc
```

### 3. **Model Download**
```bash
# Download Llama 3 8B model
python scripts/download_model.py

# Or with custom options
python scripts/download_model.py \
  --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
  --output_dir "models/Meta-Llama-3-8B-Instruct" \
  --check_space
```

## 📊 Usage Examples

### **1. Quick Start**
```bash
# Check project setup
python quick_start.py

# Run basic tests
python -m pytest tests/
```

### **2. Training**
```bash
# Start training with default settings
python run_train.py

# Custom training configuration
python run_train.py \
  --data_path "data/raw/my_data.json" \
  --output_dir "outputs/my_training" \
  --config_path "config/my_config.yaml"
```

### **3. Inference**
```bash
# Interactive mode
python run_inference.py --model_path outputs/ --interactive

# Single generation
python run_inference.py \
  --model_path outputs/ \
  --instruction "Explain quantum computing" \
  --temperature 0.8
```

### **4. Evaluation**
```bash
# Comprehensive evaluation
python scripts/evaluate.py \
  --model_path outputs/ \
  --output_dir "evaluation_results"
```

## 🔧 Configuration

### **Training Configuration** (`config/training_config.yaml`)
- Model settings and LoRA parameters
- Training hyperparameters and optimization
- Data processing and validation settings
- Output and logging configuration

### **Model Configuration** (`config/model_config.yaml`)
- Architecture details and tokenizer settings
- Hardware optimization parameters
- Quantization options (4-bit, 8-bit)

## 📈 Performance Optimization

### **Memory Management**
- **Gradient Checkpointing**: Reduces memory usage by ~30%
- **CPU Offloading**: Offloads optimizer states to CPU
- **Dynamic Memory**: Automatic memory allocation and cleanup
- **Mixed Precision**: BF16 training for speed and memory efficiency

### **Training Speed**
- **MPS Acceleration**: 2-5x faster than CPU training
- **Optimized Attention**: Flash attention when available
- **Efficient Data Loading**: Optimized data pipeline with caching

## 🧪 Testing & Validation

### **Unit Tests**
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_basic.py

# Run with coverage
python -m pytest --cov=src tests/
```

### **Integration Tests**
- Data processing pipeline validation
- Model loading and LoRA setup verification
- Training loop functionality testing
- Inference quality assessment

## 📊 Monitoring & Logging

### **Training Metrics**
- Loss curves and learning rate schedules
- Memory usage and GPU utilization
- Training speed (tokens per second)
- Validation metrics and early stopping

### **Logging Options**
- **File Logging**: Comprehensive logs in `logs/` directory
- **TensorBoard**: Real-time training visualization
- **WandB Integration**: Optional experiment tracking
- **Console Output**: Real-time progress monitoring

## 🚨 Troubleshooting

### **Common Issues**

1. **Out of Memory**
   - Reduce batch size or sequence length
   - Enable gradient checkpointing
   - Use CPU offloading for optimizer states

2. **Slow Training**
   - Verify MPS backend availability
   - Check mixed precision settings
   - Monitor CPU/GPU utilization

3. **Model Loading Issues**
   - Ensure sufficient disk space
   - Verify Hugging Face token permissions
   - Check model compatibility

### **Performance Tips**
- Start with small batch sizes and increase gradually
- Use gradient accumulation for effective larger batch sizes
- Monitor memory usage with `psutil`
- Enable mixed precision training (BF16)

## 🔮 Future Enhancements

### **Planned Features**
- **Quantization**: 4-bit and 8-bit training support
- **Distributed Training**: Multi-device training support
- **Advanced LoRA**: AdaLoRA, QLoRA variants
- **Model Compression**: Pruning and distillation
- **Web Interface**: Gradio-based training UI

### **Research Integration**
- **Custom LoRA Methods**: Novel parameter-efficient approaches
- **Advanced Optimization**: Lion, AdaFactor optimizers
- **Curriculum Learning**: Progressive difficulty training
- **Meta-Learning**: Few-shot adaptation capabilities

## 📚 Additional Resources

### **Documentation**
- **README.md**: Comprehensive project guide
- **Code Comments**: Detailed inline documentation
- **Configuration Examples**: Sample configurations for different use cases
- **Troubleshooting Guide**: Common issues and solutions

### **Community & Support**
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Community support and questions
- **Examples**: Sample training scripts and configurations
- **Contributing**: Guidelines for contributions

## 🎉 Conclusion

This project provides a robust, production-ready foundation for fine-tuning Llama 3 models on Apple Silicon Mac Minis. With its comprehensive feature set, memory optimizations, and Mac-specific enhancements, it enables efficient and effective fine-tuning workflows suitable for both research and production use.

The modular architecture makes it easy to extend and customize for specific use cases, while the comprehensive configuration system ensures flexibility and reproducibility. Whether you're a researcher exploring parameter-efficient fine-tuning or a developer building production AI applications, this project provides the tools and infrastructure needed for success.

---

**Happy Fine-tuning! 🚀**
