# MeshUnwarp

My Python desktop project for document cleanup, geometric dewarping, and text extraction. This portfolio edition contains my application source, with the original runtime code kept unchanged.

## Pipeline

1. Preprocess images with OpenCV and NumPy.
2. Assess document quality and choose normal processing or salvage behavior.
3. Use pretrained DocTr GeoTr for geometric correction and IllTr for conditional illumination correction.
4. Render Clean Scan or Ink Recovery output.
5. Extract text with **Apple Vision as the primary OCR engine on macOS**. Use **EasyOCR as the fallback** when Vision is unavailable or its output is sparse.

Tkinter provides input/output previews and processing controls. A worker process keeps image processing separate from the UI.

## Setup

Use Python 3.11 with Tkinter available:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The full application also requires a compatible `DocTr/` checkout containing `GeoTr.py`, `IllTr.py`, and their dependencies, plus these pretrained weights:

```text
DocTr/model_pretrained/geotr.pth
DocTr/model_pretrained/illtr.pth
```

The [upstream DocTr project](https://github.com/fh2019ustc/DocTr) describes the model and provides weight download links. My original local DocTr dependency was not committed. Compatibility with a newly downloaded upstream checkout, including its device handling, still needs verification. This copy does not claim to reproduce a full dewarping run without those dependencies.

After restoring them, run `python run.py` from this repository's root. Sample documents belong under `datasets/user_papers/`. The UI creates category folders when it starts.

## Verified separately

On macOS, the unchanged Stage 5 module extracted text from a synthetic document with Apple Vision. A second check, with Vision availability disabled only in the test harness, successfully used the actual EasyOCR engine. These checks verify the OCR stage, not the missing DocTr stage or accuracy on a document dataset.

The existing smoke suite can be run with `python tests/mw_smoke.py --quick` after adding sample images. Omit `--quick` only after restoring DocTr and the weights.

## Portfolio copy

This edition excludes private reports, receipt images, datasets, model weights, generated documents, personal notes, and the original Git history. My original repository remains separate. DocTr and both OCR engines are third-party components; I do not claim to have created or trained those models.
