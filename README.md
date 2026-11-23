# Single Object Tracking with a Dynamic Camera

CPU-oriented single-object tracking using background camera-motion compensation, OpenCV appearance tracking, template verification, and re-detection.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Evaluate against ground truth

Ground-truth CSV columns: `frame,x,y,w,h` (zero-based frame index; original-video pixel coordinates).

```bash
python benchmark.py --video sample.mp4 --gt ground_truth.csv --tracker CSRT --compensate
```

## Documentation

The technical report and setup guide are in `docs/`. Install documentation dependencies with:

```bash
pip install -r docs-requirements.txt
mkdocs serve
```

The GitHub Actions workflow deploys the documentation to GitHub Pages on pushes to `main`.

See the [project report](docs/index.md) and [setup guide](docs/Setup_Guide.md).
