"""Evaluate one video/tracker configuration against frame-level ground truth.

Ground-truth CSV columns: frame,x,y,w,h (zero-based frame index; pixel xywh).
Example:
  python benchmark.py --video clip.mp4 --gt annotations.csv --tracker CSRT --compensate
"""
import argparse
import time
from pathlib import Path

import cv2
import pandas as pd

from tracker import DynamicCameraTracker


def iou_xywh(a, b):
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def evaluate(video, gt_csv, tracker_name="CSRT", compensate=True,
             max_width=640, max_frames=0, output_csv=None):
    gt = pd.read_csv(gt_csv)
    required = {"frame", "x", "y", "w", "h"}
    missing = required - set(gt.columns)
    if missing:
        raise ValueError(f"Ground-truth CSV missing columns: {sorted(missing)}")
    gt = gt.drop_duplicates("frame").set_index("frame")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Video contains no readable frames.")
    scale = min(1.0, max_width / first.shape[1])
    if scale < 1:
        first = cv2.resize(first, None, fx=scale, fy=scale)
    init_gt = gt.loc[0] if 0 in gt.index else None
    if init_gt is None:
        cap.release()
        raise ValueError("Ground truth must contain frame 0 to initialize the tracker.")
    box = tuple(int(float(init_gt[k]) * scale) for k in ("x", "y", "w", "h"))
    tracker = DynamicCameraTracker(tracker_name, compensate=compensate)
    tracker.init(first, box)

    rows = []
    frame_idx = 0
    started = time.perf_counter()
    frame = first
    while True:
        if frame_idx > 0:
            ok, frame = cap.read()
            if not ok:
                break
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
        if max_frames and frame_idx >= max_frames:
            break
        if frame_idx == 0:
            result_box, status, score = tracker.img_box, "tracking", 1.0
        else:
            result = tracker.update(frame)
            result_box, status, score = result.box, result.status, result.score

        if frame_idx in gt.index:
            target = gt.loc[frame_idx]
            gt_box = [float(target[k]) * scale for k in ("x", "y", "w", "h")]
            px, py, pw, ph = map(float, result_box)
            pc = (px + pw / 2, py + ph / 2)
            gc = (gt_box[0] + gt_box[2] / 2, gt_box[1] + gt_box[3] / 2)
            cle = ((pc[0] - gc[0]) ** 2 + (pc[1] - gc[1]) ** 2) ** 0.5
            iou = iou_xywh(result_box, gt_box)
        else:
            iou, cle = float("nan"), float("nan")
        rows.append({"frame": frame_idx, "x": result_box[0], "y": result_box[1],
                     "w": result_box[2], "h": result_box[3], "status": status,
                     "score": score, "iou": iou, "cle": cle})
        frame_idx += 1

    cap.release()
    elapsed = time.perf_counter() - started
    df = pd.DataFrame(rows)
    annotated_count = int(df["iou"].notna().sum())
    summary = {
        "video": str(video), "tracker": tracker_name,
        "camera_compensation": bool(compensate), "frames_processed": len(df),
        "ground_truth_frames_matched": annotated_count,
        "mean_iou": float(df["iou"].mean()) if annotated_count else float("nan"),
        "success_rate_iou_0_5": float((df.loc[df.iou.notna(), "iou"] >= .5).mean()) if annotated_count else float("nan"),
        "mean_cle_px": float(df["cle"].mean()) if annotated_count else float("nan"),
        "lost_frame_rate": float((df["status"] == "lost").mean()) if len(df) else float("nan"),
        "tracking_retention": float((df["status"] != "lost").mean()) if len(df) else float("nan"),
        "processing_fps": len(df) / elapsed if elapsed > 0 else float("nan"),
        "video_fps": fps_in,
        "elapsed_seconds": elapsed,
    }
    if output_csv:
        df.to_csv(output_csv, index=False)
    return summary, df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--gt", required=True, help="CSV with frame,x,y,w,h")
    p.add_argument("--tracker", choices=["CSRT", "KCF", "MIL"], default="CSRT")
    p.add_argument("--compensate", action="store_true")
    p.add_argument("--max-width", type=int, default=640)
    p.add_argument("--max-frames", type=int, default=0, help="0 means all frames")
    p.add_argument("--output", default="predictions.csv")
    args = p.parse_args()
    summary, _ = evaluate(args.video, args.gt, args.tracker, args.compensate,
                          args.max_width, args.max_frames, args.output)
    print(pd.Series(summary).to_string())
    print(f"\nPer-frame predictions written to: {args.output}")


if __name__ == "__main__":
    main()
