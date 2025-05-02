#!/usr/bin/env bash
set -e

# 1. Go to the NxDT repo you already cloned
cd ~/nki-llama/fine-tune/neuronx-distributed-training

# 2. Patch train.sh only once
# --- begin patch‑once block ---------------------------------
PATCH_SENTINEL='${CONF_FILE_PATH:="./conf/${CONF_FILE}.yaml"}'

if grep -qF "$PATCH_SENTINEL" examples/train.sh ; then
    echo "✓ train.sh already patched — skipping"
else
    echo "Patching train.sh …"
    git apply - <<'DIFF'
diff --git a/examples/train.sh b/examples/train.sh
--- a/examples/train.sh
+++ b/examples/train.sh
@@
-CONF_FILE_PATH="./conf/${CONF_FILE}.yaml"
+: ${CONF_FILE_PATH:="./conf/${CONF_FILE}.yaml"}
DIFF
fi
# --- end patch‑once block -----------------------------------


# 3. Move into the examples folder
cd examples

# 4. Export config variables (order matters!)
export CONF_FILE=hf_llama3.1_8B_SFT_lora_config
export CONF_FILE_PATH="/home/ubuntu/nki-llama/fine-tune/configs/YAML/${CONF_FILE}.yaml"
export COMPILE=1      # tells train.sh to just pre‑compile graphs

# 5. Run AOT compile
bash train.sh

