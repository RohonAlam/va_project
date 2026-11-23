"""Single-object tracker for dynamic (moving) cameras. CPU only, OpenCV based.

Pipeline per frame:
  1. Estimate camera motion (LK optical flow on background points + RANSAC
     similarity transform), excluding the target region.
  2. Warp the frame into a common reference frame (stabilisation).
  3. Run an appearance tracker (CSRT/KCF/MIL) on the stabilised frame.
  4. Verify with template correlation; if lost, coast on predicted velocity and
     try to re-detect the target by template matching.
  5. Map the box back to original image coordinates.
"""
from dataclasses import dataclass
import cv2
import numpy as np


def _find_factory(name):
    for mod in (cv2, getattr(cv2, "legacy", None)):
        if mod is None:
            continue
        factory = getattr(mod, f"Tracker{name}_create", None)
        if factory is not None:
            return factory
    return None


def available_trackers():
    """Trackers present in this OpenCV build (CSRT/KCF need opencv-contrib on 4.x)."""
    return [n for n in ("CSRT", "KCF", "MIL") if _find_factory(n) is not None]


def make_cv_tracker(name="CSRT"):
    factory = _find_factory(name.upper())
    if factory is None:
        avail = available_trackers()
        if not avail:
            raise RuntimeError("No OpenCV tracker available; install opencv-contrib-python-headless")
        factory = _find_factory(avail[0])
    return factory()


def _to3(M):
    out = np.eye(3)
    out[:2] = M
    return out


def _clip_box(b, W, H):
    x, y, w, h = b
    x = int(max(0, min(W - 2, x)))
    y = int(max(0, min(H - 2, y)))
    w = int(max(2, min(W - x, w)))
    h = int(max(2, min(H - y, h)))
    return x, y, w, h


def _transform_box(box, M3):
    x, y, w, h = box
    pts = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    pts = cv2.perspectiveTransform(pts[None], M3)[0]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


@dataclass
class Result:
    frame: int
    box: tuple  # x, y, w, h in original image coordinates
    status: str  # "tracking" | "lost"
    score: float


class DynamicCameraTracker:
    def __init__(self, tracker_name="CSRT", compensate=True,
                 score_thresh=0.30, redetect_thresh=0.55, template_size=48):
        self.tracker_name = tracker_name
        self.compensate = compensate
        self.score_thresh = score_thresh
        self.redetect_thresh = redetect_thresh
        self.ts = template_size

    # ---------- helpers ----------
    def _patch(self, img, box):
        x, y, w, h = _clip_box(box, img.shape[1], img.shape[0])
        p = img[y:y + h, x:x + w]
        return cv2.resize(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), (self.ts, self.ts))

    def _score(self, img, box):
        p = self._patch(img, box)
        return float(cv2.matchTemplate(p, self.template, cv2.TM_CCOEFF_NORMED)[0, 0])

    def _camera_motion(self, prev_gray, gray, box_img):
        """3x3 matrix mapping prev-frame points to current-frame points."""
        H, W = gray.shape
        mask = np.full((H, W), 255, np.uint8)
        x, y, w, h = [int(v) for v in box_img]
        pad = int(0.25 * max(w, h))
        mask[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad] = 0
        pts = cv2.goodFeaturesToTrack(prev_gray, 400, 0.01, 8, mask=mask)
        if pts is None or len(pts) < 10:
            return np.eye(3)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
        good = st.ravel() == 1
        if good.sum() < 10:
            return np.eye(3)
        M, _ = cv2.estimateAffinePartial2D(
            pts[good], nxt[good], method=cv2.RANSAC, ransacReprojThreshold=3.0)
        return np.eye(3) if M is None else _to3(M)

    def _init_cv(self, img, box):
        self.cv = make_cv_tracker(self.tracker_name)
        self.cv.init(img, tuple(int(v) for v in _clip_box(box, img.shape[1], img.shape[0])))

    # ---------- public API ----------
    def init(self, frame, box):
        H, W = frame.shape[:2]
        self.W, self.H = W, H
        self.A = np.eye(3)               # current image -> reference coords
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.ref_box = _clip_box(box, W, H)
        self.img_box = self.ref_box
        self.template = self._patch(frame, self.ref_box)
        self.init_template = self.template.copy()
        self.vel = np.zeros(2)           # object velocity in reference coords
        self.lost_count = 0
        self.n = 0
        self._init_cv(frame, self.ref_box)

    def _redetect(self, stab):
        x, y, w, h = self.ref_box
        cx, cy = x + w / 2, y + h / 2
        r = 1.5 * max(w, h) + 15 * min(self.lost_count, 20)
        x0, y0 = int(max(0, cx - r)), int(max(0, cy - r))
        x1, y1 = int(min(self.W, cx + r)), int(min(self.H, cy + r))
        win = cv2.cvtColor(stab[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        best = (-1, None)
        for tpl in (self.template, self.init_template):
            for s in (0.9, 1.0, 1.1):
                tw = max(8, int(w * s))
                th = max(8, int(h * s))
                t = cv2.resize(tpl, (tw, th))
                if win.shape[0] <= th or win.shape[1] <= tw:
                    continue
                res = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(res)
                if mx > best[0]:
                    best = (mx, (x0 + loc[0], y0 + loc[1], tw, th))
        return best

    def update(self, frame):
        self.n += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.compensate:
            M = self._camera_motion(self.prev_gray, gray, self.img_box)
            try:
                self.A = self.A @ np.linalg.inv(M)
            except np.linalg.LinAlgError:
                pass
        self.prev_gray = gray
        stab = cv2.warpAffine(frame, self.A[:2], (self.W, self.H)) if self.compensate else frame

        old = self.ref_box
        ok, nb = self.cv.update(stab)
        score = -1.0
        if ok:
            nb = _clip_box(nb, self.W, self.H)
            score = self._score(stab, nb)
        if ok and score >= self.score_thresh:
            self.lost_count = 0
            self.ref_box = nb
            c_old = np.array([old[0] + old[2] / 2, old[1] + old[3] / 2])
            c_new = np.array([nb[0] + nb[2] / 2, nb[1] + nb[3] / 2])
            self.vel = 0.7 * self.vel + 0.3 * (c_new - c_old)
            if score > 0.8:  # slowly adapt template while confident
                self.template = cv2.addWeighted(self.template, 0.9, self._patch(stab, nb), 0.1, 0)
            status = "tracking"
        else:
            self.lost_count += 1
            x, y, w, h = old
            pred = _clip_box((x + self.vel[0] * 0.9, y + self.vel[1] * 0.9, w, h), self.W, self.H)
            self.ref_box = pred
            status = "lost"
            s, b = self._redetect(stab)
            if b is not None and s >= self.redetect_thresh:
                self.ref_box = _clip_box(b, self.W, self.H)
                self._init_cv(stab, self.ref_box)
                self.lost_count = 0
                score, status = s, "tracking"
            else:
                score = max(score, s)

        # Back to image coordinates.
        img_box = _transform_box(self.ref_box, np.linalg.inv(self.A)) if self.compensate else self.ref_box
        self.img_box = _clip_box(img_box, self.W, self.H)

        # Re-anchor reference frame if the target nears the border of the warped canvas.
        x, y, w, h = self.ref_box
        m = 0.05 * min(self.W, self.H)
        if self.compensate and (x < m or y < m or x + w > self.W - m or y + h > self.H - m):
            self.A = np.eye(3)
            self.ref_box = self.img_box
            self._init_cv(frame, self.ref_box)
        return Result(self.n, self.img_box, status, float(score))


def draw(frame, res, trail):
    x, y, w, h = res.box
    color = (0, 200, 0) if res.status == "tracking" else (0, 0, 255)
    out = frame.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    trail.append((x + w // 2, y + h // 2))
    for a, b in zip(trail[-40:-1], trail[-39:]):
        cv2.line(out, a, b, (255, 200, 0), 2)
    cv2.putText(out, f"{res.status} {res.score:.2f}", (x, max(15, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out
