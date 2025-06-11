# NKI-LLAMA: AWS Neuron Development Platform

A unified platform for fine-tuning, benchmarking, and serving LLaMA models on AWS Trainium and Inferentia using Neuron SDK's advanced optimization capabilities.

## 🎯 Overview

NKI-LLAMA provides a streamlined interface for the complete LLM development lifecycle on AWS Neuron hardware:

- **Fine-tune** models using NeuronX Distributed (NxD)
- **Optimize** with Neuron Kernel Interface (NKI) compilation
- **Benchmark** performance with comprehensive evaluation tools
- **Serve** models with vLLM's OpenAI-compatible API
- **Build** LLM-powered applications and agents

## 🔄 Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
│                 │     │                  │     │                 │     │              │
│   Fine-tuning   │────▶│ NKI Compilation  │────▶│ vLLM Inference  │────▶│    Agent     │
│      (NxD)      │     │  & Benchmarking  │     │     (NxDI)      │     │ Development  │
│                 │     │                  │     │                 │     │              │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └──────────────┘
        │                         │                         │                         │
        ▼                         ▼                         ▼                         ▼
  Trained Model            NKI-Optimized              API Endpoint              LLM Apps
                          Model Artifacts            (OpenAI Compatible)
```

### Key Technologies

- **NKI (Neuron Kernel Interface)**: Custom kernel optimizations for AWS Neuron
- **NxD (NeuronX Distributed)**: Distributed training framework
- **NxDI (NeuronX Distributed Inference)**: Optimized inference runtime
- **vLLM**: High-performance serving with Neuron backend

## 📋 Requirements

### System Requirements
- **Instance**: trn1.32xlarge (recommended)
- **AMI**: Deep Learning AMI Neuron (Ubuntu 22.04)
- **Neuron SDK**: 2.23.0
- **Python**: 3.10

### SDK Components
- NeuronX Distributed Training: 1.3.0
- NeuronX Distributed Inference: 0.3.5591
- Neuron Compiler: 2.18.121.0

## 🚀 Quick Start

### 1. Instance Setup
```bash
# Create EC2 instance
# - Type: trn1.32xlarge
# - AMI: Deep Learning AMI Neuron (Ubuntu 22.04)
# - Storage: 512GB+ recommended
```

### 2. Installation
```bash
# Clone repository
git clone https://github.com/your-org/nki-llama.git
cd nki-llama

# Install
chmod +x install.sh
./install.sh

# Configure
cp .env.example .env
nano .env  # Add your HF_TOKEN
```

### 3. First Run
```bash
# Interactive setup
./nki-llama setup

# Download model
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate
./nki-llama inference download

# Run benchmark (compiles model on first run)
tmux new -s benchmark
./nki-llama inference benchmark
```

## 💻 Command Reference

### Core Commands

| Command | Description |
|---------|-------------|
| `./nki-llama setup` | Interactive setup wizard |
| `./nki-llama status` | System and project status |
| `./nki-llama config` | Display configuration |
| `./nki-llama clean` | Clean artifacts and cache |

### Fine-tuning Pipeline

```bash
# Activate environment
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate

# Complete pipeline
./nki-llama finetune all

# Or run individual steps
./nki-llama finetune deps      # Install dependencies
./nki-llama finetune data      # Download dataset
./nki-llama finetune model     # Download base model
./nki-llama finetune convert   # Convert to NxDT format
./nki-llama finetune compile   # Pre-compile graphs
./nki-llama finetune train     # Start training
```

### Benchmarking & Compilation

```bash
# Activate environment
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate

# Download model (if not already done)
./nki-llama inference download

# Full benchmark with NKI compilation (default)
./nki-llama inference benchmark

# Benchmark with options
./nki-llama inference benchmark --seq-len 1024
./nki-llama inference benchmark --clear-cache  # Clear compilation cache
./nki-llama inference benchmark --no-nki       # Without NKI optimizations
```

#### Benchmark Modes

| Mode | Description | Status |
|------|-------------|--------|
| `evaluate_all` | Full benchmark with NKI compilation and caching | ✅ Working |
| `evaluate_single` | Quick validation test | ⚠️ Not implemented |

> **Note**: The `evaluate_single` mode is currently not functional. Use `evaluate_all` (default) for all benchmarking needs.

### Inference Serving

```bash
# Setup vLLM (one-time)
./nki-llama inference setup

# Start API server
tmux new -s vllm
./nki-llama inference server

# Server will use NKI-compiled artifacts from benchmarking
```

### Development Tools

```bash
# Start Jupyter Lab
./nki-llama jupyter

# Access at http://your-instance-ip:8888
```

## 🛠️ Advanced Usage

### Cache Management

The compilation cache can accumulate failed entries. Monitor and manage it:

```bash
# Check cache status
./nki-llama status

# Clear cache before benchmark
./nki-llama inference benchmark --clear-cache

# Manual cache cleanup
./nki-llama clean
```

### Using tmux (Recommended)

Long-running operations should use tmux to prevent disconnection issues:

```bash
# Create session
tmux new -s session-name

# Run command
./nki-llama [command]

# Detach: Ctrl+B, then D

# List sessions
tmux ls

# Reattach
tmux attach -t session-name
```

### Environment Management

Different operations require specific environments:

```bash
# Fine-tuning
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate

# Inference & Benchmarking
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate

# Agent Development
source ~/nki-llama/venv/bin/activate
```

## 📊 Monitoring & Debugging

### System Monitoring
```bash
# Neuron device status
neuron-ls

# Real-time usage
neuron-top

# Project status
./nki-llama status
```

### Log Files
```bash
# View recent logs
ls -la logs/
tail -f logs/nki-llama_*.log

# Benchmark results
cat logs/benchmarks/*/metadata.json
```

### Common Issues

#### Compilation Cache Errors
```bash
# Symptoms: "Got a cached failed neff" errors
# Solution:
./nki-llama inference benchmark --clear-cache
```

#### SIGHUP Errors
```bash
# Symptoms: Process terminated during compilation
# Solution: Always use tmux for long operations
tmux new -s benchmark
```

#### Memory Issues
```bash
# Monitor memory usage
neuron-top

# Adjust parallelism if needed
export TENSOR_PARALLEL_SIZE=4  # Reduce from 8
```

## 🏗️ Project Structure

```
nki-llama/
├── nki-llama.sh          # Main CLI interface
├── nki-llama.config      # System configuration
├── .env                  # User configuration
├── install.sh            # Installation script
├── README.md             # This file
├── src/
│   ├── fine-tune/        # Training pipeline
│   │   └── scripts/      # Training automation
│   └── inference/        # Inference pipeline
│       ├── main.py       # Benchmark entry point
│       └── scripts/      # Inference automation
├── notebooks/            # Example notebooks
│   └── travel_agent.ipynb
├── logs/                 # Operation logs
│   └── benchmarks/       # Benchmark results
└── models/              # Downloaded models
    └── compiled/        # NKI-compiled artifacts
```

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Hugging Face Access
HF_TOKEN=your_token_here

# Model Selection
MODEL_ID=meta-llama/Meta-Llama-3-8B
MODEL_NAME=llama-3-8b

# Hardware Configuration
TENSOR_PARALLEL_SIZE=8
NEURON_RT_NUM_CORES=32

# Training Parameters
BATCH_SIZE=1
MAX_STEPS=1000
SEQ_LENGTH=2048
LEARNING_RATE=5e-5

# Inference Parameters
INFERENCE_PORT=8080
MAX_MODEL_LEN=2048
```

## 🎓 Complete Workflow Example

### Step 1: Fine-tune a Model
```bash
tmux new -s training
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate
./nki-llama finetune all
# Detach: Ctrl+B, D
```

### Step 2: Benchmark & Compile
```bash
tmux new -s benchmark
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate
./nki-llama inference download
./nki-llama inference benchmark
# First run compiles with NKI (10-30 minutes)
# Detach: Ctrl+B, D
```

### Step 3: Serve Model
```bash
tmux new -s server
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate
./nki-llama inference server
# API available at http://localhost:8080
# Detach: Ctrl+B, D
```

### Step 4: Build Applications
```bash
# Terminal 1: Keep server running
# Terminal 2: Development
./nki-llama jupyter
# Open browser to http://your-ip:8888
```

## 📚 Additional Resources

- [AWS Neuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)
- [NeuronX Distributed Training Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/index.html)
- [NKI Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html)
- [vLLM Neuron Integration](https://docs.vllm.ai/en/latest/getting_started/neuron-installation.html)

## 🐛 Known Issues

- **evaluate_single mode**: Currently not implemented. Use default `evaluate_all` mode for all benchmarking.
- **First compilation**: Initial NKI compilation can take 10-30 minutes. Subsequent runs use cache.
- **Cache corruption**: If benchmark fails with cache errors, use `--clear-cache` flag.

## 📄 License

© 2025 Amazon Web Services. All rights reserved.

This project is provided under the AWS Customer Agreement and integrates with AWS Neuron SDK components subject to their respective licenses.