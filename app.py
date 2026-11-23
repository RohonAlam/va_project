import os
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
import pandas as pd
import streamlit as st

from tracker import DynamicCameraTracker, available_trackers, draw

st.set_page_config(page_title="Dynamic Camera Object Tracker", layout="wide")

# Keep the playback player compact on both desktop and mobile layouts.
st.markdown("""
<style>
/* Streamlit's video element otherwise expands to the full column width. */
div[data-testid="stVideo"] {
    width: min(100%, 420px) !important;
    max-width: 420px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
div[data-testid="stVideo"] video {
    width: 100% !important;
    max-width: 420px !important;
    max-height: 260px !important;
    height: auto !important;
    object-fit: contain !important;
}
</style>
""", unsafe_allow_html=True)


def read_first_frame(path):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def run_tracking(path, box, tracker_name, compensate, max_width, max_frames, progress):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
    ok, frame = cap.read()
    scale = min(1.0, max_width / frame.shape[1])
    if scale < 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
        box = tuple(int(v * scale) for v in box)
    H, W = frame.shape[:2]

    raw_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    tracker = DynamicCameraTracker(tracker_name, compensate=compensate)
    tracker.init(frame, box)
    rows, trail = [], []
    writer.write(draw(frame, type("R", (), dict(box=tracker.img_box, status="tracking", score=1.0))(), trail))
    rows.append(dict(frame=0, x=box[0], y=box[1], w=box[2], h=box[3], status="tracking", score=1.0))

    n = 1
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        res = tracker.update(frame)
        writer.write(draw(frame, res, trail))
        x, y, w, h = res.box
        rows.append(dict(frame=n, x=x, y=y, w=w, h=h, status=res.status, score=round(res.score, 3)))
        n += 1
        if n % 5 == 0:
            progress.progress(min(1.0, n / max(1, min(total, max_frames))))
    cap.release()
    writer.release()
    progress.progress(1.0)

    # Re-encode to H.264 so the browser can play it.
    out_path = raw_path.replace(".mp4", "_h264.mp4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", raw_path, "-vcodec", "libx264",
                    "-pix_fmt", "yuv420p", "-loglevel", "error", out_path], check=True)
    return out_path, pd.DataFrame(rows), fps, scale


st.title("Single Object Tracking with a Dynamic Camera")
st.caption("Camera-motion compensation (optical flow + RANSAC) + CSRT appearance tracking + template re-detection. CPU only.")

with st.sidebar:
    st.header("Settings")
    tracker_name = st.selectbox("Appearance tracker", available_trackers(), index=0)
    compensate = st.toggle("Camera motion compensation", value=True)
    max_width = st.select_slider("Processing width (px)", [480, 640, 800, 960, 1280], value=640)
    max_frames = st.number_input("Max frames to process", 50, 5000, 600, step=50)

uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
gt_file = st.file_uploader("Optional ground-truth CSV (frame,x,y,w,h)", type=["csv"])
if uploaded:
    if st.session_state.get("_name") != uploaded.name:
        tmp = tempfile.NamedTemporaryFile(suffix=os.path.splitext(uploaded.name)[1], delete=False)
        tmp.write(uploaded.getbuffer())
        tmp.close()
        st.session_state["_name"], st.session_state["_path"] = uploaded.name, tmp.name
    path = st.session_state["_path"]
    first = read_first_frame(path)
    if first is None:
        st.error("Could not read the video.")
        st.stop()
    H, W = first.shape[:2]

    st.subheader("1. Select the target on the first frame")
    c1, c2 = st.columns([1, 2])
    with c1:
        left = st.slider("Left (%)", 0.0, 100.0, 40.0, 0.5)
        top = st.slider("Top (%)", 0.0, 100.0, 40.0, 0.5)
        width = st.slider("Width (%)", 1.0, 100.0, 15.0, 0.5)
        height = st.slider("Height (%)", 1.0, 100.0, 15.0, 0.5)
    box = (int(left / 100 * W), int(top / 100 * H), int(width / 100 * W), int(height / 100 * H))
    preview = first.copy()
    cv2.rectangle(preview, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
    with c2:
        st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption=f"Frame size {W}x{H}", use_container_width=True)

    st.subheader("2. Track")
    if st.button("Run tracking", type="primary"):
        bar = st.progress(0.0)
        try:
            with st.spinner("Tracking..."):
                out, df, fps, scale = run_tracking(
                    path, box, tracker_name, compensate, max_width, int(max_frames), bar
                )
            st.session_state["result"] = (out, df, fps, scale)
        except Exception as exc:
            st.error(f"Tracking failed: {exc}")
            st.exception(exc)

    if "result" in st.session_state:
        out, df, fps, scale = st.session_state["result"]
        # CSS above enforces a fixed maximum player width even on wide layouts.
        st.video(out)
        lost = int((df.status == "lost").sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Frames processed", len(df))
        m2.metric("Frames flagged lost", lost)
        m3.metric("Frames with confident track", f"{1 - lost / len(df):.1%}")
        if gt_file is not None:
            try:
                gt = pd.read_csv(gt_file)
                required = {"frame", "x", "y", "w", "h"}
                if not required.issubset(gt.columns):
                    st.error("Ground-truth CSV must contain: frame,x,y,w,h")
                else:
                    merged = df.merge(gt[list(required)], on="frame", how="inner", suffixes=("_pred", "_gt"))
                    if not merged.empty:
                        for key in ("x", "y", "w", "h"):
                            merged[f"{key}_gt"] = merged[f"{key}_gt"].astype(float) * scale
                        x1 = merged[["x_pred", "x_gt"]].max(axis=1)
                        y1 = merged[["y_pred", "y_gt"]].max(axis=1)
                        x2 = pd.concat([merged.x_pred + merged.w_pred, merged.x_gt + merged.w_gt], axis=1).min(axis=1)
                        y2 = pd.concat([merged.y_pred + merged.h_pred, merged.y_gt + merged.h_gt], axis=1).min(axis=1)
                        inter = (x2 - x1).clip(lower=0) * (y2 - y1).clip(lower=0)
                        union = merged.w_pred * merged.h_pred + merged.w_gt * merged.h_gt - inter
                        merged["iou"] = inter / union.replace(0, float("nan"))
                        pcx = merged.x_pred + merged.w_pred / 2
                        pcy = merged.y_pred + merged.h_pred / 2
                        gcx = merged.x_gt + merged.w_gt / 2
                        gcy = merged.y_gt + merged.h_gt / 2
                        merged["cle_px"] = ((pcx - gcx) ** 2 + (pcy - gcy) ** 2) ** 0.5
                        e1, e2, e3 = st.columns(3)
                        e1.metric("Mean IoU", f"{merged.iou.mean():.3f}")
                        e2.metric("Success Rate @ 0.50", f"{(merged.iou >= 0.5).mean():.1%}")
                        e3.metric("Mean CLE (px)", f"{merged.cle_px.mean():.1f}")
                        st.caption(f"Metrics computed over {len(merged)} frames with matching annotations. GT coordinates are assumed to refer to the original video resolution.")
                    else:
                        st.warning("No frame indices overlap between predictions and ground truth.")
            except Exception as exc:
                st.error(f"Could not evaluate ground truth: {exc}")
        st.dataframe(df, use_container_width=True, height=250)
        d1, d2 = st.columns(2)
        d1.download_button("Download trajectory CSV", df.to_csv(index=False), "trajectory.csv", "text/csv")
        with open(out, "rb") as f:
            d2.download_button("Download annotated video", f.read(), "tracked.mp4", "video/mp4")
else:
    st.info("Upload a video to begin.")
