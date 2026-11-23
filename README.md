# Single Object Tracking with a Dynamic Camera

CPU-only tracker built with OpenCV: camera-motion compensation (LK optical flow + RANSAC
similarity transform) -> frame stabilisation -> CSRT appearance tracking -> template-based
re-detection after loss. Streamlit front end in `app.py`.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
1. Push this folder to a GitHub repo (`app.py` and `requirements.txt` at the repo root).
2. Open https://share.streamlit.io, sign in with GitHub, choose "Create app",
   select the repo/branch and set the main file to `app.py`. (UI labels may have changed; check current docs.)

## Files
| File | Purpose |
|---|---|
| `tracker.py` | Tracking pipeline (`DynamicCameraTracker`) |
| `app.py` | Streamlit UI: upload, select box, run, download video/CSV |
| `requirements.txt` | Dependencies |
