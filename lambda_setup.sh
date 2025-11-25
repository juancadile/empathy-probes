#!/bin/bash
# Lambda Labs Instance Setup Script
# Run this on each Lambda instance after SSH connection

set -e

echo "=================================="
echo "LAMBDA INSTANCE SETUP"
echo "=================================="
echo ""

# Check GPU
echo "Checking GPU..."
nvidia-smi
echo ""

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get install -y git python3-pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install vllm torch transformers accelerate

# Create working directory
echo "Setting up working directory..."
mkdir -p ~/empathy-probes
cd ~/empathy-probes

# Clone repository (or create structure)
echo "Setting up project structure..."
mkdir -p data/eia_scenarios
mkdir -p data/contrastive_pairs
mkdir -p src

echo ""
echo "=================================="
echo "SETUP COMPLETE"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Upload scenarios.json to ~/empathy-probes/data/eia_scenarios/"
echo "2. Upload generate_opensource_vllm.py to ~/empathy-probes/src/"
echo "3. Run generation script"
echo ""
