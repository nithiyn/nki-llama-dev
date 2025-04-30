#!/usr/bin/env bash
set -e

DATA_DIR=./dataset
mkdir -p ${DATA_DIR}

# Download Dolly SFT dataset from S3 (no sign-request)
aws s3 cp s3://neuron-s3/training_datasets/llama/sft/training.jsonl ${DATA_DIR}/training.jsonl --no-sign-request
aws s3 cp s3://neuron-s3/training_datasets/llama/sft/validation.jsonl ${DATA_DIR}/validation.jsonl --no-sign-request
