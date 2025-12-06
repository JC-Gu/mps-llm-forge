# Setup Guide: Virtual Environment & Dependencies

## 🐍 Virtual Environment Setup

This guide will help you set up the complete development environment for the Llama 3 LoRA Fine-tuning project.

## 📋 Prerequisites

- **macOS 12.3+** (for MPS support)
- **Python 3.8+** (Python 3.13 recommended)
- **20GB+ free disk space** for models and dependencies
- **16GB+ RAM** (32GB+ recommended)

## 🚀 Quick Setup

### 1. **Clone and Navigate**
```bash
git clone <repository-url>
cd mps-llm-forge
```

### 2. **Create Virtual Environment**
```bash
python3 -m venv venv
```

### 3. **Activate Virtual Environment**
```bash
source venv/bin/activate
```

### 4. **Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. **Verify Setup**
```bash
python quick_start.py
```

## 🔧 Detailed Setup Steps

### **Step 1: Environment Creation**
```bash
# Create virtual environment
python3 -m venv venv

# Verify creation
ls -la venv/
```

### **Step 2: Activation**
```bash
# Activate virtual environment
source venv/bin/activate

# Verify activation (should show venv in prompt)
which python
# Should show: /path/to/mps-llm-forge/venv/bin/python
```

### **Step 3: Dependency Installation**
```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Verify key packages
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import peft; print('PEFT:', peft.__version__)"
```

### **Step 4: Verification**
```bash
# Run quick start script
python quick_start.py

# Run tests
python -m pytest tests/ -v
```

## 📦 Package Management

### **Core Dependencies**
- **PyTorch 2.8.0+** - Deep learning framework with MPS support
- **Transformers 4.56.0+** - Hugging Face model library
- **PEFT 0.17.1+** - Parameter-efficient fine-tuning
- **Accelerate 1.10.1+** - Distributed training utilities
- **Datasets 4.0.0+** - Data loading and processing

### **Data & Visualization**
- **Pandas 2.3.2+** - Data manipulation
- **Matplotlib 3.10.6+** - Plotting and visualization
- **Seaborn 0.13.2+** - Statistical visualization
- **Scikit-learn 1.7.1+** - Machine learning utilities

### **Monitoring & Logging**
- **TensorBoard 2.20.0+** - Training visualization
- **WandB 0.21.3+** - Experiment tracking
- **PSUtil 7.0.0+** - System monitoring

### **Utilities**
- **PyYAML 6.0.2+** - Configuration file parsing
- **TQDM 4.67.1+** - Progress bars
- **SentencePiece 0.2.1+** - Tokenization
- **Protobuf 6.32.0+** - Serialization

## 🔍 Environment Verification

### **Hardware Check**
```bash
# Check MPS availability
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# Check device info
python -c "from src.model_utils import get_device_info; print(get_device_info())"
```

### **Package Verification**
```bash
# Check all installed packages
pip list

# Check specific package versions
python -c "
import torch, transformers, peft, accelerate
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'Accelerate: {accelerate.__version__}')
"
```

### **Project Structure Check**
```bash
# Run quick start verification
python quick_start.py

# Run unit tests
python -m pytest tests/ -v
```

## 🚨 Troubleshooting

### **Common Issues**

#### 1. **Virtual Environment Not Found**
```bash
# Error: No module named 'venv'
# Solution: Install venv module
python3 -m pip install --user virtualenv
python3 -m virtualenv venv
```

#### 2. **Permission Denied**
```bash
# Error: Permission denied when creating venv
# Solution: Check directory permissions
ls -la
chmod 755 .
```

#### 3. **Package Installation Failures**
```bash
# Error: Failed to install package
# Solution: Upgrade pip and try again
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
```

#### 4. **MPS Not Available**
```bash
# Error: MPS backend not available
# Solution: Check macOS version and PyTorch installation
sw_vers  # Should show 12.3+
python -c "import torch; print(torch.backends.mps.is_available())"
```

#### 5. **Memory Issues**
```bash
# Error: Out of memory during installation
# Solution: Install packages one by one
pip install torch
pip install transformers
pip install peft
# ... continue for other packages
```

### **Performance Optimization**

#### **Installation Speed**
```bash
# Use faster package index
pip install -i https://pypi.org/simple/ -r requirements.txt

# Use local cache
pip install --cache-dir ~/.pip/cache -r requirements.txt
```

#### **Memory Usage**
```bash
# Install packages with minimal dependencies
pip install --no-deps torch
pip install --no-deps transformers
# ... continue for other packages
```

## 🔄 Environment Management

### **Activation Scripts**
```bash
# Use the provided activation script
./activate_env.sh

# Or create an alias in your shell profile
echo 'alias activate_llama="cd /path/to/mps-llm-forge && source venv/bin/activate"' >> ~/.zshrc
source ~/.zshrc
```

### **Deactivation**
```bash
# Deactivate virtual environment
deactivate

# Verify deactivation
which python  # Should show system Python
```

### **Environment Updates**
```bash
# Activate environment
source venv/bin/activate

# Update packages
pip install --upgrade -r requirements.txt

# Or update specific packages
pip install --upgrade torch transformers peft
```

### **Clean Reinstall**
```bash
# Remove virtual environment
rm -rf venv

# Recreate environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 📊 Environment Monitoring

### **System Resources**
```bash
# Check disk space
df -h

# Check memory usage
free -h  # Linux
vm_stat   # macOS

# Check Python process
ps aux | grep python
```

### **Package Sizes**
```bash
# Check package sizes
pip show torch transformers peft
du -sh venv/lib/python*/site-packages/torch
du -sh venv/lib/python*/site-packages/transformers
```

## 🎯 Next Steps After Setup

1. **Set Hugging Face Token**
   ```bash
   export HF_TOKEN="your_token_here"
   ```

2. **Download Model**
   ```bash
   python scripts/download_model.py
   ```

3. **Prepare Training Data**
   ```bash
   # Your data should be in data/raw/ directory
   # Supported formats: JSON, CSV, TXT
   ```

4. **Start Training**
   ```bash
   python src/training.py
   ```

5. **Run Inference**
   ```bash
   python src/inference.py --model_path outputs/ --interactive
   ```

## 📚 Additional Resources

- **README.md** - Comprehensive project documentation
- **PROJECT_SUMMARY.md** - Detailed project overview
- **requirements.txt** - Package dependencies
- **requirements_exact.txt** - Exact package versions for reproducibility

## 🆘 Getting Help

If you encounter issues:

1. **Check the troubleshooting section above**
2. **Run verification scripts**: `python quick_start.py`
3. **Check system requirements**: macOS 12.3+, Python 3.8+
4. **Verify package versions**: `pip list`
5. **Check hardware compatibility**: MPS support

---

**Happy Setup! 🚀**

Your virtual environment is now ready for Llama 3 LoRA fine-tuning on Mac Mini!
