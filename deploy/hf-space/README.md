# Hugging Face Space Helper

This folder prepares the Assignment 1 image demo for a Hugging Face Gradio Space.

## Do you need another project folder?

No. If you already ran:

```powershell
git clone https://huggingface.co/spaces/BoChay/DeepLearning
```

then that cloned `DeepLearning` folder is already the deploy repository.

## What this helper does

The sync script copies only the files needed by the Space:

- `app.py` at the Space root
- `requirements.txt`
- `.gitattributes` for Git LFS
- `assignments/assignment-1/app/`
- `assignments/assignment-1/image/artifacts/`
- `assignments/assignment-1/image/models/`
- `assignments/assignment-1/image/data/cifar-10-python.tar.gz`

The sync step removes calibration `.png` files in the Space repo. The app can
rebuild those plots from the exported JSON metrics, which avoids binary-file
push issues on Hugging Face Spaces.

## Recommended workflow

1. Make sure the Hugging Face Space repo is cloned somewhere on your machine.
2. Run the sync script from this repository:

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\hf-space\sync_to_space.ps1 -SpaceRepoPath "C:\path\to\DeepLearning"
```

3. Move into the cloned Space repo:

```powershell
cd C:\path\to\DeepLearning
```

4. Enable Git LFS and push:

```powershell
git lfs install
git add .
git commit -m "Add Gradio image demo"
git push
```

## Result

After the push, Hugging Face Spaces will build the app from:

- `app.py`
- `requirements.txt`

The Gradio app will then load the copied Assignment 1 image assets from
`assignments/assignment-1/`.
