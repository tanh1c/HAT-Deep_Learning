#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/HAT-Deep_Learning}"
MINICONDA_DIR="${MINICONDA_DIR:-/root/miniconda3}"
ENV_NAME="${ENV_NAME:-dl}"
SESSION_NAME="${SESSION_NAME:-jupyter}"
PORT="${PORT:-8888}"

export PATH="${MINICONDA_DIR}/bin:${PATH}"
eval "$("${MINICONDA_DIR}/bin/conda" shell.bash hook)"
conda activate "${ENV_NAME}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists."
else
  tmux new-session -d -s "${SESSION_NAME}" \
    "bash -lc 'source /root/.bashrc >/dev/null 2>&1 || true; eval \"\$(${MINICONDA_DIR}/bin/conda shell.bash hook)\"; conda activate \"${ENV_NAME}\"; cd \"${REPO_DIR}\"; jupyter lab --ip 127.0.0.1 --port ${PORT} --no-browser --allow-root'"
  sleep 3
fi

echo
echo "Jupyter session started in tmux: ${SESSION_NAME}"
echo
echo "Recent Jupyter output:"
tmux capture-pane -pt "${SESSION_NAME}" | tail -n 30 || true
echo
echo "Attach to the session with:"
echo "  tmux attach -t ${SESSION_NAME}"
echo
echo "From your local Windows machine, create the SSH tunnel with:"
echo "  ssh -L ${PORT}:127.0.0.1:${PORT} root@<YOUR_VM_PUBLIC_IP>"
echo
