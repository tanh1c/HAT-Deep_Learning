#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/tanh1c/HAT-Deep_Learning.git}"
REPO_DIR="${REPO_DIR:-/root/HAT-Deep_Learning}"
MINICONDA_DIR="${MINICONDA_DIR:-/root/miniconda3}"
ENV_NAME="${ENV_NAME:-dl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "Please run this script as root."
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

echo "==> Updating system packages"
apt update
apt upgrade -y
apt install -y git curl wget tmux htop unzip build-essential python3-pip ca-certificates

echo "==> GPU sanity check"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found. Check the selected VM image."
fi

echo "==> Installing Miniconda if needed"
if [[ ! -x "${MINICONDA_DIR}/bin/conda" ]]; then
  TMP_INSTALLER="/tmp/miniconda.sh"
  wget -O "${TMP_INSTALLER}" "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  bash "${TMP_INSTALLER}" -b -p "${MINICONDA_DIR}"
fi

export PATH="${MINICONDA_DIR}/bin:${PATH}"

if ! grep -q "${MINICONDA_DIR}/bin" /root/.bashrc 2>/dev/null; then
  echo "export PATH=${MINICONDA_DIR}/bin:\$PATH" >> /root/.bashrc
fi

"${MINICONDA_DIR}/bin/conda" init bash >/dev/null 2>&1 || true
eval "$("${MINICONDA_DIR}/bin/conda" shell.bash hook)"

echo "==> Accepting Conda Terms of Service if required"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

echo "==> Creating environment ${ENV_NAME} if needed"
if ! conda info --envs | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

echo "==> Cloning or updating repository"
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
else
  git -C "${REPO_DIR}" pull --ff-only || true
fi

cd "${REPO_DIR}"

echo "==> Upgrading pip"
python -m pip install --upgrade pip

echo "==> Installing PyTorch with CUDA wheels"
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "==> Installing notebook and analysis dependencies"
python -m pip install numpy pandas matplotlib seaborn scikit-learn scipy tqdm pillow jupyterlab ipywidgets

echo "==> Python / CUDA check"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

cat <<EOF

Setup completed.

Next steps on the VM:

  bash ${REPO_DIR}/start_jupyter_vm.sh

Then on your local Windows machine:

  ssh -L 8888:127.0.0.1:8888 root@<YOUR_VM_PUBLIC_IP>

Open in your browser:

  http://127.0.0.1:8888/lab

If you need the full manual guide, see:

  ${REPO_DIR}/FPT_GPU_VM_SETUP.md

EOF
