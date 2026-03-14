# HAT Deep Learning Course Repository

This repository stores both the GitHub Pages site and the working code for the course assignments.

## Repository layout

```text
docs/
  GitHub Pages site

assignments/
  assignment-1/
    app/
    image/
    text/
    multimodal/
    shared/
```

## Current focus

- `docs/` contains the public course site and assignment pages.
- `assignments/assignment-1/` contains the working area for Assignment 1.
- The `image/` track is already populated with notebooks, reports, artifacts, and the app integration.
- `text/` and `multimodal/` now have placeholders so teammates can add their work cleanly.

## Running the local Assignment 1 app

```powershell
python assignments/assignment-1/app/main.py
```

The app expects local image checkpoints inside `assignments/assignment-1/image/models/`.

## Notes before pushing

- `assignments/assignment-1/image/data/` is ignored because datasets should stay out of Git.
- `*.pth` is ignored because model checkpoints should use Git LFS or external storage.
- `docs/` is ready to publish with GitHub Pages using the `/docs` folder.
