#!/bin/bash
# reasoning-bench-lm-eval.sh - Start vLLM OpenAI-compatible API server and run lm-eval

set -euo pipefail

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../nki-llama.config"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Setting up vLLM for Neuron...${NC}"

# Check if in correct environment
if [[ "$VIRTUAL_ENV" != *"inference"* ]]; then
    echo -e "${RED}Error: Not in inference environment${NC}"
    echo -e "Run: source ${NEURON_INFERENCE_VENV}/bin/activate"
    exit 1
fi

# Clone or update vLLM repository
if [[ -d "$VLLM_REPO" ]]; then
    echo "Updating existing vLLM repository..."
    cd "$VLLM_REPO"
    git fetch
    git pull
else
    echo "Cloning vLLM repository..."
    cd "$(dirname "$VLLM_REPO")"
    git clone https://github.com/vllm-project/vllm.git
fi

# Install requirements
cd "$VLLM_REPO"
echo "Installing vLLM requirements..."
pip install -U -r requirements/neuron.txt

# Install vLLM
echo "Installing vLLM for Neuron..."
VLLM_TARGET_DEVICE="neuron" pip install -e .

# Ensure transformers < 4.50 (needed by Neuron hf_adapter)
python - <<'PY'
import subprocess, pkg_resources, sys
req = "4.50.0"
try:
    ver = pkg_resources.get_distribution("transformers").version
except pkg_resources.DistributionNotFound:
    ver = ""
if not ver or pkg_resources.parse_version(ver) >= pkg_resources.parse_version(req):
    print("Installing transformers<%s …" % req)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"transformers<{req}"])
PY

echo -e "${GREEN}✓ vLLM setup complete${NC}"

cd "$HOME"
git clone https://github.com/aws-neuron/aws-neuron-samples.git
cd /home/ubuntu/aws-neuron-samples/inference-benchmarking/
pip install -r requirements.txt --quiet

echo -e "${GREEN}✓ Inference-Benchmarking setup complete${NC}"

#write config file for reasoning test
cd /home/ubuntu/aws-neuron-samples/inference-benchmarking/

if test -f "/home/ubuntu/aws-neuron-samples/inference-benchmarking/reasoning_bench.yaml"; then
   echo "config file exists."
else 
    echo "Creating config file..."
fi

OUT_FILE="reasoning_bench.yaml" 
cat > "$OUT_FILE" <<YAML

server:
  name: "Reasoning-benchmark server"
  model_path: "${NKI_MODELS}/${MODEL_NAME}"
  model_s3_path: null
  compiled_model_path: "${NKI_COMPILED}/${MODEL_NAME}"
  max_seq_len: ${MAX_MODEL_LEN}
  context_encoding_len: ${MAX_MODEL_LEN}
  tp_degree: ${TENSOR_PARALLEL_SIZE}
  n_vllm_threads: ${TENSOR_PARALLEL_SIZE}
  server_port: ${INFERENCE_PORT}
  continuous_batch_size: 1

test:
  accuracy:
    mytest:
      client: "lm_eval"
      datasets: ["gsm8k_cot", "mmlu_flan_n_shot_generative_logical_fallacies"]
      max_concurrent_requests: 1
      timeout: 3600
      client_params:
        limit: 200
        use_chat: True

YAML

#config file written
echo -e "${GREEN}✓ Config File written${NC}"

#run reasoning benchmark
echo -e "${GREEN}Starting Reasoning Benchmarking job...${NC}"
echo $"{BLUE}----- reasoning_bench.yaml -----${NC}"
cat reasoning_bench.yaml
echo

python accuracy.py --config reasoning_bench.yaml