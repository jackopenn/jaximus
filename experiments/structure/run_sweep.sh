#!/bin/bash

# Configuration
ZONE="${ZONE:-us-central2-b}"
NODE="${NODE:-node-69}"
PROJECT="${PROJECT:-trm-jax}"

# Required env vars
: "${WANDB_API_KEY:?Set WANDB_API_KEY}"
: "${HF_TOKEN:?Set HF_TOKEN}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMD_FILE="${SCRIPT_DIR}/sweep_cmds.txt"

# Filter out comments and empty lines, then run each command
while read -r args; do
    echo "Running: $args"
    gcloud compute tpus tpu-vm ssh --zone "$ZONE" "$NODE" --project "$PROJECT" --worker=all --command \
        "cd jaximus && git checkout sliding-window-exp && git pull && source .venv/bin/activate && WANDB_API_KEY=$WANDB_API_KEY PYTHONPATH=. JAX_MULTIHOST=1 HF_TOKEN=$HF_TOKEN python train.py --config experiments/structure/config.py $args" </dev/null || echo "Failed: $args"
done < <(grep -v '^#' "$CMD_FILE" | grep -v '^$')
