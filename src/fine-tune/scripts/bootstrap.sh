#!/usr/bin/env bash
set -e

echo "==== Starting NeuronX setup script ===="

# 1. Activate your Neuron venv
#source /opt/aws_neuronx_venv_pytorch_2_5/bin/activate
echo "==== Neuron venv activation commented out - uncomment if needed ===="

# 2. Upgrade pip, neuron packages active cause venv enabled, cython error - ensure installed
echo "==== Upgrading pip and installing prerequisites ===="
pip install -U pip
echo "==== Installing Cython ===="
pip install Cython
echo "==== Upgrading setuptools ===="
pip install --upgrade setuptools

# 3. Build a slim CPU-only Apex (NeMo dependency)
echo "==== Checking for Apex repository ===="
if [ ! -d "apex" ]; then
    echo "==== Cloning Apex repository ===="
    git clone https://github.com/NVIDIA/apex.git apex
else
    echo "==== Apex repository already exists, skipping clone ===="
fi

cd apex
echo "==== Checking out Apex version 23.05 ===="
git checkout 23.05

# 3a. Overwrite setup.py with the nxd specific setup
echo "==== Creating custom setup.py for CPU-only Apex ===="
cat > setup.py <<'EOF'
import sys
import warnings
import os
from packaging.version import parse, Version

from setuptools import setup, find_packages
import subprocess

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME, load

setup(
    name="apex",
    version="0.1",
    packages=find_packages(
        exclude=("build", "csrc", "include", "tests", "dist", "docs", "tests", "examples", "apex.egg-info",)
    ),
    install_requires=["packaging>20.6"],
    description="PyTorch Extensions written by NVIDIA",
)
EOF

# 3b. Install build prerequisites and make the wheel
echo "==== Installing packaging and wheel for building Apex ===="
pip install packaging wheel
echo "==== Building Apex wheel ===="
python setup.py bdist_wheel

# Copy out the generated wheel path for later
echo "==== Saving Apex wheel path ===="
APEX_WHEEL=$(ls ~/nki-llama/src/fine-tune/apex/dist/apex-*.whl | head -1)
echo "Apex wheel: $APEX_WHEEL"

# 3c. Return to root of our example
echo "==== Returning to script directory ===="
cd ..

# 4. Install the rest of NxDT's Python requirements
#    (this includes NeMo, PyTorch-Lightning, etc.)
echo "==== Downloading NxDT requirements.txt ===="
wget -q https://raw.githubusercontent.com/aws-neuron/neuronx-distributed-training/master/requirements.txt
echo "==== Installing NxDT dependencies ===="
pip install -r requirements.txt

# 5. Finally, install our freshly built Apex wheel
echo "==== Installing Apex wheel ===="
pip install "$APEX_WHEEL"

# 6. And install the NxDT training framework itself
echo "==== Installing neuronx_distributed_training package ===="
pip install neuronx_distributed_training --extra-index-url https://pip.repos.neuron.amazonaws.com

echo "==== Setup complete! ===="