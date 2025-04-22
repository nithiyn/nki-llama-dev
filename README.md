# NKI Llama

## Technical Infrastructure

### Compute Resources
- **Required Instance**: trn1.2xlarge
- **Base AMI**: Deep Learning AMI Neuron (Ubuntu 22.04)
- **Base Packages**:
  - NxD (NeuronX Distributed Training)
  - NKI (Neuron Kernel Interface)

## Setup Steps

1. Create a Trainium instance with AWS Neuron SDK v2.21 using EC2 with the following settings:
    1. **Name:** nki-llama
    2. **AMI:** Deep Learning AMI Neuron (Ubuntu 22.04)
    3. **Instance type:** trn1.2xlarge
    4. **Key pair (login):** create a new key pair
    5. When connecting to these instances via SSH, use the username of *ubuntu*.

2. Clone this repository and navigate to it:
   ```bash
   git clone [REPO_URL]
   cd [PATH]/nki-llama
   ```

3. Create your `.env` file by copying the provided example:
   ```bash
   cp .env.example .env
   # Edit .env file with your preferred settings
   nano .env
   ```

4. Use our Makefile to simplify the setup and execution process for building agents and applications over neuron:
   
   ```bash
   # First, activate the AWS Neuron environment
   source /opt/aws_neuronx_venv_pytorch_2_5_nxd_inference/bin/activate
   
   # Setup vLLM for Neuron
   make setup-vllm
   
   # Download the model from Hugging Face (you'll need a HF token)
   make download
   
   # Create a Python virtual environment and setup Jupyter (in a new terminal)
   # First create and activate the virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Then run the Jupyter setup target
   make setup-jupyter
   
   # Start the vLLM OpenAI-compatible API server (in first terminal with Neuron environment)
   make start-server
   
   # Start Jupyter Lab (in second terminal with the venv environment)
   make jupyter
   ```

## Environment Configuration

The repository includes a `.env.example` file with template configuration. Copy this file to create your own `.env`:

```
# Model configuration
MODEL_NAME=llama-3.2-3b-instruct

# Server configuration
PORT=8080
MAX_MODEL_LEN=2048
TENSOR_PARALLEL_SIZE=8
```

The Makefile will automatically load this configuration if present, or prompt you for values if not set.

## Running Inference

The Makefile provides several commands for running inference and evaluation:

```bash
# Environment variables will be loaded from .env automatically
make infer

# Run in evaluate_single mode
make evaluate
```

## Agent Development

This repository includes support for building LLM-powered agents using LangGraph and LangChain. A sample travel planning agent is included that demonstrates how to build a stateful agent workflow with the following capabilities:

- Context-aware travel itinerary generation
- Multi-turn conversation with memory
- Dynamic workflow management using LangGraph
- Integration with VLLMOpenAI for efficient inference on Trainium

### Jupyter Notebook

The repository includes a Jupyter notebook for developing and testing agents. To use it:

1. Ensure you've started the vLLM server in one terminal: `make start-server`
2. Start Jupyter Lab in another terminal: `make jupyter`
3. Open the travel planning notebook and select the "neuron_agents" kernel

## Makefile Commands

| Command | Description |
|---------|-------------|
| `source /opt/aws_neuronx_venv_pytorch_2_5_nxd_inference/bin/activate` | Activate AWS Neuron environment |
| `python -m venv venv && source venv/bin/activate` | Create and activate local Python virtual environment |
| `make setup-jupyter` | Install requirements and setup Jupyter kernel |
| `make setup-vllm` | Setup vLLM for Neuron |
| `make download` | Download model from Hugging Face |
| `make infer` | Run inference in generate mode |
| `make evaluate` | Run inference in evaluate_all mode |
| `make start-server` | Start vLLM OpenAI-compatible API server |
| `make jupyter` | Run Jupyter Lab server |
| `make clean` | Remove generated files |


---

© 2025 Amazon Web Services. All rights reserved.
