# FPT GPU VM Setup Guide

This guide covers the full workflow for using an FPT GPU VM with this project:

1. Connect to the VM with SSH
2. Update the machine and install core tools
3. Install and initialize Miniconda
4. Create the Python environment
5. Clone this repository
6. Install PyTorch and notebook dependencies
7. Run JupyterLab on the VM
8. Connect to JupyterLab from your local machine through `localhost`
9. Download trained `.pth` files and artifacts back to your local machine
10. Delete the VM to stop billing

## Assumptions

- VM OS: Ubuntu 24.04
- Username: `root`
- Public IP example: `124.197.18.91`
- Local machine: Windows with PowerShell and OpenSSH
- Project repo: `https://github.com/tanh1c/HAT-Deep_Learning.git`

Replace the example IP with your actual VM public IP when needed.

## 1. Connect to the VM

On your local Windows machine, open PowerShell:

```powershell
ssh root@124.197.18.91
```

If SSH does not work:

- Check the VM `Security group`
- Make sure `TCP 22` is open

## 2. Update the VM and install core tools

Run these commands on the VM:

```bash
apt update && apt upgrade -y
apt install -y git curl wget tmux htop unzip build-essential python3-pip
nvidia-smi
nvcc --version
```

Notes:

- `nvidia-smi` verifies that the GPU is visible
- `nvcc --version` checks CUDA tooling

## 3. Install Miniconda

Run on the VM:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3
echo 'export PATH=/root/miniconda3/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

Initialize conda for bash:

```bash
/root/miniconda3/bin/conda init bash
source ~/.bashrc
```

If `conda activate` still fails in the current session:

```bash
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
```

## 4. Accept Conda Terms of Service and create the environment

Run on the VM:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -y -n dl python=3.11
conda activate dl
```

Verify:

```bash
conda info --envs
which python
python --version
```

Expected Python path:

```bash
/root/miniconda3/envs/dl/bin/python
```

## 5. Clone the repository

Run on the VM:

```bash
cd /root
git clone https://github.com/tanh1c/HAT-Deep_Learning.git
cd /root/HAT-Deep_Learning
```

If the repo already exists:

```bash
cd /root/HAT-Deep_Learning
git pull
```

## 6. Install PyTorch and notebook dependencies

Activate the environment first:

```bash
conda activate dl
```

Install PyTorch with CUDA wheels:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install the rest of the notebook stack:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy tqdm pillow jupyterlab ipywidgets
```

Verify GPU access from Python:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 7. Run JupyterLab on the VM

It is best to run Jupyter inside `tmux` so it stays alive if your SSH session drops.

Start a new `tmux` session:

```bash
tmux new -s jupyter
```

Inside the `tmux` session:

```bash
conda activate dl
cd /root/HAT-Deep_Learning
jupyter lab --ip 127.0.0.1 --port 8888 --no-browser --allow-root
```

Jupyter will print a URL like:

```text
http://127.0.0.1:8888/lab?token=...
```

To detach from `tmux` and keep Jupyter running:

- Press `Ctrl+B`
- Then press `D`

To reattach later:

```bash
tmux attach -t jupyter
```

## 8. Open JupyterLab locally through `localhost`

Open a second PowerShell window on your local machine and create an SSH tunnel:

```powershell
ssh -L 8888:127.0.0.1:8888 root@124.197.18.91
```

Then open your browser on your local machine:

```text
http://127.0.0.1:8888/lab
```

If prompted, paste the token from the JupyterLab log on the VM.

## 9. Open the project notebooks

Inside JupyterLab, open:

```text
assignments/assignment-1/image/notebooks/
```

Useful notebooks:

- `stanforddogs_resnet18_vit_report_workflow.ipynb`
- `gtsrb_resnet18_vit_report_workflow.ipynb`
- `fgvcaircraft_resnet18_vit_report_workflow.ipynb`
- `oxfordiiitpet_resnet18_vit_report_workflow.ipynb`

## 10. Download `.pth` files and artifacts back to your local machine

### Option A: compress on the VM first

Example for Stanford Dogs outputs.

On the VM:

```bash
cd /root
tar -czf stanford_dogs_outputs.tar.gz \
  -C /root/HAT-Deep_Learning \
  assignments/assignment-1/image/models/stanforddogs_resnet18.pth \
  assignments/assignment-1/image/models/stanforddogs_vit_b16.pth \
  assignments/assignment-1/image/artifacts/stanford_dogs
```

On your local Windows machine:

```powershell
scp root@124.197.18.91:/root/stanford_dogs_outputs.tar.gz "C:\Users\LG\Desktop\Study Material\DL\Assignment1\"
```

Extract locally:

```powershell
cd "C:\Users\LG\Desktop\Study Material\DL\Assignment1"
tar -xzf .\stanford_dogs_outputs.tar.gz
```

### Option B: copy files directly

On your local Windows machine:

```powershell
scp root@124.197.18.91:/root/HAT-Deep_Learning/assignments/assignment-1/image/models/stanforddogs_resnet18.pth "C:\Users\LG\Desktop\Study Material\DL\Assignment1\assignments\assignment-1\image\models\"
scp root@124.197.18.91:/root/HAT-Deep_Learning/assignments/assignment-1/image/models/stanforddogs_vit_b16.pth "C:\Users\LG\Desktop\Study Material\DL\Assignment1\assignments\assignment-1\image\models\"
scp -r root@124.197.18.91:/root/HAT-Deep_Learning/assignments/assignment-1/image/artifacts/stanford_dogs "C:\Users\LG\Desktop\Study Material\DL\Assignment1\assignments\assignment-1\image\artifacts\"
```

## 11. Recommended notebook workflow

When the notebook has been edited locally and re-uploaded or re-pulled, do this in Jupyter:

1. `Kernel -> Restart Kernel`
2. Run cells from top to bottom
3. Avoid jumping into the middle of the notebook before metadata and config cells have run

This avoids stale-state errors such as missing columns in `meta`.

## 12. Common issues

### `CondaError: Run 'conda init' before 'conda activate'`

Run:

```bash
/root/miniconda3/bin/conda init bash
source ~/.bashrc
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate dl
```

### `EnvironmentNameNotFound: Could not find conda environment: dl`

Create the environment:

```bash
conda create -y -n dl python=3.11
conda activate dl
```

### `Running as root is not recommended`

Use:

```bash
jupyter lab --ip 127.0.0.1 --port 8888 --no-browser --allow-root
```

### Notebook still shows old code after edits

Do:

1. Save the notebook file
2. Close and reopen the notebook tab
3. Restart the kernel
4. Run from the top again

## 13. Stop billing on FPT GPU VM

Important:

- If your VM uses local NVMe / ephemeral storage, `Power off` may still continue billing
- To stop being charged, you usually need to **Delete the VM**

Before deleting:

1. Download all `.pth` files
2. Download all `artifacts`
3. Download any updated notebooks or logs

Then:

1. Go to the FPT AI / GPU VM portal
2. Find your VM
3. Open the `...` actions menu
4. Choose `Delete`
5. Confirm deletion

## 14. Quick start summary

### On the VM

```bash
apt update && apt upgrade -y
apt install -y git curl wget tmux htop unzip build-essential python3-pip
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3
echo 'export PATH=/root/miniconda3/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
/root/miniconda3/bin/conda init bash
source ~/.bashrc
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -y -n dl python=3.11
conda activate dl
cd /root
git clone https://github.com/tanh1c/HAT-Deep_Learning.git
cd /root/HAT-Deep_Learning
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install numpy pandas matplotlib seaborn scikit-learn scipy tqdm pillow jupyterlab ipywidgets
tmux new -s jupyter
conda activate dl
cd /root/HAT-Deep_Learning
jupyter lab --ip 127.0.0.1 --port 8888 --no-browser --allow-root
```

### On your local Windows machine

```powershell
ssh -L 8888:127.0.0.1:8888 root@124.197.18.91
```

Open:

```text
http://127.0.0.1:8888/lab
```
