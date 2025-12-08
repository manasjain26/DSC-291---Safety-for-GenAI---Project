#!/bin/bash

################################################################################
# QUICKSTART SCRIPT
# DSC291 - Prompt Injection Defense Project
#
# This script helps you get started quickly with the implementation.
################################################################################

echo "========================================================================"
echo "DSC291 Prompt Injection Defense - Quick Start"
echo "========================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the DSC291 directory"
    exit 1
fi

echo "Step 1: Checking Python environment..."
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "✅ Found Python $PYTHON_VERSION"
echo ""

echo "Step 2: Installing dependencies..."
echo "This may take a few minutes..."
pip install -q -r requirements.txt
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "⚠️  Some dependencies failed to install. Please check manually."
fi
echo ""

echo "Step 3: Creating directory structure..."
mkdir -p data/jailbreak data/benign checkpoints results logs
echo "✅ Directories created"
echo ""

echo "Step 4: Checking for Tinker API key..."
if [ -z "$TINKER_API_KEY" ]; then
    echo "⚠️  TINKER_API_KEY not found"
    echo "   The implementation will use local training as fallback."
    echo "   To use Tinker, set: export TINKER_API_KEY=your_key_here"
else
    echo "✅ TINKER_API_KEY found"
fi
echo ""

echo "========================================================================"
echo "Setup Complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo ""
echo "Option A - Run full pipeline (recommended for first time):"
echo "   python run_pipeline.py"
echo ""
echo "Option B - Run step by step:"
echo "   1. Download data:    python download_data.py"
echo "   2. Prepare data:     python prepare_training_data.py"
echo "   3. Train model:      python train_sft_tinker.py"
echo "   4. Evaluate:         python evaluate.py --model_path checkpoints/sft_lora"
echo ""
echo "Option C - Quick test with existing model:"
echo "   python quick_test.py"
echo ""
echo "For more information, see README.txt or IMPLEMENTATION_SUMMARY.txt"
echo "========================================================================"

