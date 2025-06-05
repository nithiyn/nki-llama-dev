# pipeline.sh 

---

## 1 · Prerequisites

| Requirement | Reason | Install / Notes |
|-------------|--------|-----------------|
| **Neuron virtual‑env** | Script refuses to run outside it | `source /opt/aws_neuronx_venv_pytorch_2_5/bin/activate` |
| **`scripts/` folder** | step-by step scripts for running fine tuning | |
| **`.env` file** *(optional)* | Central place for env vars | Place at `../../.env` |

Example `.env`:

```dotenv
HF_TOKEN=hf_****************************************
MODEL_ID=meta-llama-3-8b
```

---

## 2 · Setup

```bash
# Clone repo and enter it
cd ./src/fine-tune

# Make the script executable
chmod +x pipeline.sh

# Activate Neuron environment
source /opt/aws_neuronx_venv_pytorch_2_5/bin/activate
```

---

## 3 · Usage

| Command | Action |
|---------|--------|
| `./pipeline.sh` | Run the **full pipeline** (deps → data → model → convert_ckpt → precompile → train) |
| `./pipeline.sh deps` | Install/validate Apex, NxDT, etc. |
| `./pipeline.sh data` | Download dataset |
| `./pipeline.sh model` | Download & convert model checkpoints |
| `./pipeline.sh convert_ckpt` | Convert checkpoints to NxDT format |
| `./pipeline.sh precompile` | Ahead‑of‑time graph compilation |
| `./pipeline.sh train` | Start fine‑tuning |
| `./pipeline.sh clean` | Remove generated datasets, weights, experiments |

 Each sub‑command double‑checks you’re inside a Neuron venv and prints a helpful error if not.

---

## 4 · Environment Variables

| Variable | Purpose | How to set |
|----------|---------|-----------|
| `HF_TOKEN` | Hugging Face auth token (for private models) | Add to `.env` or `export HF_TOKEN=…` |
| `MODEL_ID` | Model slug, e.g. `meta-llama-3-8b` | Same as above |

The script auto‑loads `../../.env` with `set -a; source …`. Modify the `ENV_FILE=` line in `pipeline.sh` if you store it elsewhere.

---

## 5 · Troubleshooting

| Symptom | Probable Cause | Fix |
|---------|---------------|-----|
| `Not inside a Neuron virtual environment.` | Forgot to activate venv | `source /opt/aws_neuronx_venv_pytorch_2_5/bin/activate` |
| `command not found: pipeline.sh` | File not executable or wrong cwd | `chmod +x pipeline.sh` and/or `./pipeline.sh` |
| Model download fails | Missing/invalid `HF_TOKEN` | Provide valid token in env or `.env` |
| Long compile times | First‑time Neuron AOT | Subsequent runs reuse cached graphs |

---

## 6 · Extending the Pipeline

1. Add a new Bash function in `pipeline.sh` (e.g., `evaluate()`).
2. Append its name to the pattern list inside `main()`.
3. Optionally call it from `all()` for automatic inclusion.

```bash
./pipeline.sh train 
```
---
