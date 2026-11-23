# Setup & Installation Guide

## 1. Requirements

- Python 3.10 or 3.11 recommended
- pip and virtual-environment support
- OpenCV-contrib (installed through `requirements.txt`)
- FFmpeg executable bundled through `imageio-ffmpeg` for browser-compatible output encoding

The tracker runs on CPU. No CUDA toolkit or model weights are required.

## 2. Clone and install

```bash
git clone https://github.com/RohonAlam/va_project.git
cd va_project

python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you already installed another OpenCV wheel, remove conflicting OpenCV packages before reinstalling. The `opencv-python`, `opencv-contrib-python`, and their headless variants should not be installed together in the same environment.

## 3. Launch Streamlit

```bash
streamlit run app.py
```

1. Upload an MP4, AVI, MOV, or MKV video.
2. Define the initial target box using the percentage sliders.
3. Select an available appearance tracker.
4. Enable or disable camera-motion compensation.
5. Choose processing width and maximum frame count.
6. Optionally upload a ground-truth CSV with `frame,x,y,w,h`.
7. Run tracking, inspect the annotated video and trajectory table, and download the outputs.

Ground-truth coordinates must correspond to the original video dimensions. Frame indices are zero-based. The initial frame (0) must be annotated for benchmark evaluation.

## 4. Run quantitative evaluation

Example:

```bash
python benchmark.py --video sample.mp4 --gt ground_truth.csv \
  --tracker CSRT --compensate --max-width 640 \
  --output predictions.csv
```

For the baseline without compensation:

```bash
python benchmark.py --video sample.mp4 --gt ground_truth.csv \
  --tracker CSRT --max-width 640 --output baseline.csv
```

To compare KCF or MIL, change the `--tracker` argument. `--max-frames` can restrict the run; zero means process the complete video.

The output CSV contains frame index, predicted `xywh` box, status, correlation score, IoU, and CLE for frames with annotations. Summary metrics are printed to the terminal.

## 5. Ground-truth annotation format

```csv
frame,x,y,w,h
0,120,80,64,90
1,123,81,64,90
2,126,82,64,90
```

All coordinates are pixels in the original video resolution. Annotate the same frames for each ablation run. If annotations are sparse, metrics are computed only on frame indices that have ground truth.

## 6. Deploy the Streamlit application

The app can be deployed using a Streamlit-compatible hosting service by connecting the GitHub repository and setting `app.py` as the entry point. Confirm the host's current resource, upload-size, and execution-time limits before deployment. Long videos may exceed hosted CPU or memory limits; local execution is preferable for large benchmark runs.

## 7. Publish documentation with GitHub Pages

The repository includes `mkdocs.yml`, the `docs/` directory, and `.github/workflows/docs.yml`.

1. Push the files to the repository's `main` branch.
2. Open **Settings → Pages** in GitHub.
3. If prompted, configure Pages to allow deployment from the `gh-pages` branch.
4. The GitHub Actions workflow builds the MkDocs site and publishes it to `gh-pages`.
5. Check the Actions tab for build/deployment errors.

Expected site URL:

[https://rohonalam.github.io/va_project/](https://rohonalam.github.io/va_project/)

The first deployment may take a few minutes. If the workflow is blocked by repository Pages settings, adjust the Pages source as indicated by the Actions log.

## 8. Reproducibility checklist

For each reported experiment, record:

- Video name, duration, resolution, and source FPS
- Ground-truth annotation protocol and annotated frame count
- Tracker type and camera-motion compensation setting
- Processing width and frame limit
- CPU model, RAM, Python version, and OpenCV version
- Mean IoU, success rate, CLE, lost-frame rate, and processing FPS

Never publish illustrative or placeholder numbers as measured experimental results.
