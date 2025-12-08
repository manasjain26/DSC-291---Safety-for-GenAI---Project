#!/bin/bash

# Setup script for DSC291 Project
# Creates environment and installs dependencies

echo "Setting up DSC291 Prompt Injection Defense Project..."

# Create directories
mkdir -p data/jailbreak
mkdir -p data/benign
mkdir -p checkpoints
mkdir -p logs
mkdir -p results

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete! Make sure to export your TINKER_API_KEY if using Tinker:"
echo "export TINKER_API_KEY=your_api_key_here"

