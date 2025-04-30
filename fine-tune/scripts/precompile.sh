#!/usr/bin/env bash
#!/usr/bin/env bash
set -e

# 1. Clone and cd into the repo
git clone https://github.com/aws-neuron/neuronx-distributed-training.git
cd neuronx-distributed-training

# 2. Create & apply the diff
cat << 'EOF' > train-conf-file-path.patch
*** Begin Patch
*** Update File: examples/train.sh
@@
- CONF_FILE_PATH="./conf/${CONF_FILE}.yaml"
+: ${CONF_FILE_PATH:="./conf/${CONF_FILE}.yaml"}
*** End Patch
EOF

# 3a. If you have 'patch':
patch -p1 < train-conf-file-path.patch

# 3b. Or if you prefer 'git apply':
# git apply train-conf-file-path.patch

rm train-conf-file-path.patch

echo "✓ examples/train.sh patched to allow CONF_FILE_PATH override"


cd neuronx-distributed-training
# Point to our config path and file
export CONF_FILE=hf_llama3.1_8B_SFT_lora_config
export CONF_FILE=hf_llama3_8B_SFT_lora_config
export COMPILE=1

# Run AOT compile
bash train.sh
