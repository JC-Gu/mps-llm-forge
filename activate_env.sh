#!/bin/bash
# Virtual Environment Activation Script for Llama 3 LoRA Fine-tuning Project

echo "🐍 Activating Virtual Environment for Llama 3 LoRA Fine-tuning"
echo "=============================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment 'venv' not found!"
    echo "Please run: python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "✅ Activating virtual environment..."
source venv/bin/activate

# Check Python version and packages
echo ""
echo "📊 Environment Information:"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"
echo "Working directory: $(pwd)"

# Check key packages
echo ""
echo "📦 Key Package Versions:"
echo "PyTorch: $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not installed")"
echo "Transformers: $(python -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "Not installed")"
echo "PEFT: $(python -c "import peft; print(peft.__version__)" 2>/dev/null || echo "Not installed")"
echo "Accelerate: $(python -c "import accelerate; print(accelerate.__version__)" 2>/dev/null || echo "Not installed")"

# Check MPS availability
echo ""
echo "🍎 Mac Mini Hardware Check:"
if python -c "import torch; print('MPS available:', torch.backends.mps.is_available())" 2>/dev/null; then
    echo "✅ PyTorch MPS backend is available"
else
    echo "❌ PyTorch MPS backend not available"
fi

echo ""
echo "🚀 Virtual environment is now active!"
echo "To deactivate, run: deactivate"
echo ""
echo "Next steps:"
echo "1. Set your Hugging Face token: export HF_TOKEN='your_token_here'"
echo "2. Download the model: python scripts/download_model.py"
echo "3. Start training: python run_train.py"
echo ""
echo "💡 Tip: You can also run 'python quick_start.py' to verify everything is working!"
