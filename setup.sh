#!/usr/bin/env bash
set -e

# Update package lists
sudo apt-get update

# Install system dependencies
sudo apt-get install -y python3.9 python3.9-venv python3.9-dev bzip2 tmux git git-lfs libglu1-mesa-dev

# Create and activate virtual environment
python3.9 -m venv /content/py39
source /content/py39/bin/activate

# Upgrade pip
pip install --upgrade pip==24.0

# Install Python dependencies
#pip install -r ./pre-requirements.txt
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r ./requirements.txt
git lfs install
git clone https://huggingface.co/H-Liu1997/emage_evaltools
