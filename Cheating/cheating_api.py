import cv2
import mediapipe as mp
import numpy as np
import datetime
import time
import os
import sys
import json
import uuid
import shutil
import threading
from collections import deque
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# ─────────────────────────────────────────────────────────────
# Suppress TensorFlow / MediaPipe noise
# ─────────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel']      = '3'
sys.stderr = open(os.devnull, 'w')
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────
# SCREEN RECORDING IMPORTS
# ─────────────────────────────────────────────────────────────
try:
    import mss
    import mss.tools
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print("[WARN] mss not installed — screen recording disabled. Run: pip install mss")

try:
    import pygetwindow as gw
    PYGETWINDOW_AVAILABLE = True
except ImportError:
    PYGETWINDOW_AVAILABLE = False
    print("[WARN] pygetwindow not installed — window-title tracking disabled. Run: pip install pygetwindow")

# ─────────────────────────────────────────────────────────────
# DIRECTORIES
# ─────────────────────────────────────────────────────────────
BASE_DIR    = Path(r"D:\VVSPL\AI Study Planner\Cheating")
UPLOAD_DIR  = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# DETECTION CONFIGURATION
# ─────────────────────────────────────────────────────────────
BLINK_CLOSED_THRESHOLD   = 4.0
PARTIAL_BLINK_THRESHOLD  = 5.5
MAX_CLOSED_FRAMES        = 2
MAX_EYE_CLOSED_DURATION  = 3.0
GAZE_SUSTAINED_FRAMES    = 6
GAZE_UP_MAX_DURATION     = 3.0
HEAD_PITCH_LIMIT         = 25
HEAD_YAW_LIMIT           = 30
HEAD_ROLL_LIMIT          = 20
HEAD_POSE_FRAMES         = 8
HEAD_YAW_PARTIAL_LIMIT   = 15
HEAD_YAW_PARTIAL_FRAMES  = 12
MOUTH_OPEN_THRESHOLD     = 10.0
MOUTH_PARTIAL_THRESHOLD  = 5.0
MOUTH_OPEN_FRAMES        = 12
MOUTH_PARTIAL_FRAMES     = 20
MOUTH_MOVEMENT_WINDOW    = 5.0
MOUTH_MOVEMENT_THRESHOLD = 0.40
RAPID_BLINK_WINDOW       = 10.0
RAPID_BLINK_COUNT        = 8
SCREENSHOT_COOLDOWN      = 2.0
ATTENTION_DECAY          = 0.995
ATTENTION_PENALTY        = 5
BG_MIN_AREA              = 6000
BG_MIN_HEIGHT_RATIO      = 0.20
BG_SUBJECT_ZONE          = 0.50
BG_WARMUP_FRAMES         = 120
BG_MOTION_CONFIRM        = 4
BODY_CHECK_INTERVAL      = 0.2
EXTRA_BODY_FRAMES        = 4

# ── Tab-switch / Screen-change configuration ──────────────────
SCREEN_SAMPLE_INTERVAL   = 0.5    # seconds between screen polls
SCREEN_DIFF_THRESHOLD    = 0.12   # fraction of pixels that must change
SCREEN_DIFF_CONFIRM      = 2      # consecutive diff samples before flagging
SCREEN_CROP_MARGIN       = 60     # edge pixels to ignore (taskbar/clock)
TAB_SWITCH_COOLDOWN      = 3.0    # seconds between tab-switch alerts
MONITOR_INDEX            = 1      # 1 = primary monitor

# ── Screenshot overlay settings ───────────────────────────────
OVERLAY_FONT_SCALE  = 0.85
OVERLAY_THICKNESS   = 2
OVERLAY_PADDING     = 14
OVERLAY_BOX_COLOR   = (0, 0, 160)       # dark red banner (BGR)
OVERLAY_TEXT_COLOR  = (255, 255, 255)
OVERLAY_SUB_COLOR   = (210, 210, 210)

LEFT_EYE_LM     = [159, 145, 33, 133, 468]
RIGHT_EYE_LM    = [386, 374, 263, 362, 473]
LEFT_BLINK_LM   = [159, 145]
RIGHT_BLINK_LM  = [386, 374]
MOUTH_TOP_LM    = 13
MOUTH_BOTTOM_LM = 14
HEAD_LM_IDS     = [1, 152, 263, 33, 287, 57]
HEAD_3D_POINTS  = np.array([
    (0.0,    0.0,    0.0),
    (0.0,  -330.0,  -65.0),
    (-225.0, 170.0, -135.0),
    ( 225.0, 170.0, -135.0),
    (-150.0,-150.0, -125.0),
    ( 150.0,-150.0, -125.0),
], dtype=np.float64)

# ─────────────────────────────────────────────────────────────
# JOB REGISTRY
# ─────────────────────────────────────────────────────────────
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────
# SCREEN MONITOR
# ─────────────────────────────────────────────────────────────

class ScreenMonitor:
    """
    Watches the machine's screen while a video is being analysed.
    On every confirmed tab/window switch it saves TWO full-resolution
    screenshots:

        cheat_tab_switch_*_BEFORE.png  — screen just before the switch
        cheat_tab_switch_*_AFTER.png   — screen just after the switch

    Both images carry a red alert banner showing the reason and timestamp.

    Detection methods
    ─────────────────
    Method A – Window-title tracking (Windows only, needs pygetwindow)
        Reads the active foreground window title every SCREEN_SAMPLE_INTERVAL
        seconds. Any title change triggers a save. If exam_window_hint is set,
        only switches *away* from that keyword are marked as off-task.

    Method B – Pixel-diff fallback (cross-platform, needs mss)
        Compares consecutive 320×180 thumbnails of the cropped screen.
        A diff_ratio above SCREEN_DIFF_THRESHOLD for SCREEN_DIFF_CONFIRM
        consecutive samples flags a sudden scene change.
    """

    def __init__(self, job_id: str, screenshot_dir: Path,
                 event_log: list, exam_window_hint: str = ""):
        self.job_id          = job_id
        self.screenshot_dir  = screenshot_dir
        self.event_log       = event_log
        self.exam_hint       = exam_window_hint.lower()

        self._stop_event     = threading.Event()
        self._thread         = threading.Thread(target=self._run, daemon=True)

        # Title-tracking state
        self._last_title     = ""

        # Pixel-diff state
        self._prev_thumb_gray: np.ndarray | None = None   # 320×180 grayscale
        self._prev_fullres:    np.ndarray | None = None   # native-res BGR

        self._diff_streak         = 0
        self._last_alert_time: float = 0.0
        self.tab_switch_count     = 0

    # ── public ───────────────────────────────────────────────
    def start(self): self._thread.start()
    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=5)

    # ── screen capture ───────────────────────────────────────
    def _grab_fullres(self) -> np.ndarray | None:
        """Capture the primary monitor at native resolution → BGR ndarray."""
        if not MSS_AVAILABLE:
            return None
        try:
            with mss.mss() as sct:
                mon = sct.monitors[MONITOR_INDEX]
                raw = sct.grab(mon)
                img = np.frombuffer(raw.bgra, dtype=np.uint8).reshape(
                    (raw.height, raw.width, 4))
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception:
            return None

    def _thumb_from_fullres(self, fullres: np.ndarray) -> np.ndarray | None:
        """Derive a 320×180 grayscale thumbnail, cropping taskbar margins."""
        if fullres is None:
            return None
        m    = SCREEN_CROP_MARGIN
        h, w = fullres.shape[:2]
        crop = fullres[m:h-m, m:w-m] if (2*m < h and 2*m < w) else fullres
        return cv2.cvtColor(cv2.resize(crop, (320, 180)), cv2.COLOR_BGR2GRAY)

    # ── annotation ───────────────────────────────────────────
    def _annotate(self, frame: np.ndarray,
                  tag: str, reason: str, ts: str) -> np.ndarray:
        """
        Draw a two-line alert banner at the top of a full-res screenshot.

        Line 1 (large, white):   "⚠  TAB SWITCH DETECTED  [BEFORE]"
        Line 2 (smaller, grey):  reason text + timestamp
        """
        out  = frame.copy()
        h, w = out.shape[:2]
        pad  = OVERLAY_PADDING
        fs   = OVERLAY_FONT_SCALE
        th   = OVERLAY_THICKNESS
        font = cv2.FONT_HERSHEY_SIMPLEX

        line1 = f"TAB SWITCH DETECTED  [{tag}]"
        line2 = f"{reason[:100]}   {ts}"

        (_, l1h), _ = cv2.getTextSize(line1, font, fs,       th)
        (_, l2h), _ = cv2.getTextSize(line2, font, fs * 0.7, th - 1)

        box_h = l1h + l2h + pad * 3
        cv2.rectangle(out, (0, 0), (w, box_h), OVERLAY_BOX_COLOR, -1)

        cv2.putText(out, line1,
                    (pad, pad + l1h),
                    font, fs, OVERLAY_TEXT_COLOR, th, cv2.LINE_AA)
        cv2.putText(out, line2,
                    (pad, pad + l1h + pad + l2h),
                    font, fs * 0.7, OVERLAY_SUB_COLOR, th - 1, cv2.LINE_AA)
        return out

    # ── save BEFORE + AFTER pair ─────────────────────────────
    def _save_pair(self,
                   before: np.ndarray | None,
                   after:  np.ndarray | None,
                   reason: str,
                   key: str) -> bool:
        """
        Write two annotated PNG files and update the job registry.
        Returns False if within the cooldown window (nothing saved).
        """
        now = time.time()
        if now - self._last_alert_time < TAB_SWITCH_COOLDOWN:
            return False
        self._last_alert_time = now
        self.tab_switch_count += 1

        ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        saved     = []
        blank     = np.zeros((1080, 1920, 3), dtype=np.uint8)   # fallback

        for tag, frame in [("BEFORE", before), ("AFTER", after)]:
            src = frame if frame is not None else blank
            ann = self._annotate(src, tag, reason, ts)
            fn  = str(self.screenshot_dir / f"cheat_{key}_{ts}_{tag}.png")
            cv2.imwrite(fn, ann)
            saved.append(fn)

        self.event_log.append({
            "time"  : ts,
            "reason": reason,
            "type"  : "tab_switch",
            "files" : saved,          # ["...BEFORE.png", "...AFTER.png"]
        })

        with jobs_lock:
            if self.job_id in jobs:
                jobs[self.job_id]["tab_switch_count"] = self.tab_switch_count
                jobs[self.job_id]["cheat_count"] = (
                    jobs[self.job_id].get("cheat_count", 0) + 1)
        return True

    # ── Method A: window-title ───────────────────────────────
    def _check_title(self, before: np.ndarray | None,
                     after: np.ndarray | None) -> bool:
        if not PYGETWINDOW_AVAILABLE:
            return False
        try:
            active = gw.getActiveWindow()
            title  = active.title.strip() if active else ""
        except Exception:
            return False

        if title == self._last_title:
            return False

        prev, self._last_title = self._last_title, title
        if not prev:                        # first sample — record baseline
            return False

        is_off = self.exam_hint and (self.exam_hint not in title.lower())
        prefix = "[OFF-TASK] " if is_off else ""
        reason = f"{prefix}Window: '{prev[:40]}' → '{title[:40]}'"

        return self._save_pair(before, after, reason, "tab_switch_title")

    # ── Method B: pixel-diff ─────────────────────────────────
    def _check_pixel_diff(self, before: np.ndarray | None,
                          after_thumb: np.ndarray | None,
                          after_full:  np.ndarray | None) -> bool:
        if after_thumb is None:
            return False

        if self._prev_thumb_gray is None:
            self._prev_thumb_gray = after_thumb
            return False

        diff       = cv2.absdiff(after_thumb,
                                 self._prev_thumb_gray).astype(np.float32)
        diff_ratio = float(np.mean(diff > 30))
        self._prev_thumb_gray = after_thumb

        if diff_ratio > SCREEN_DIFF_THRESHOLD:
            self._diff_streak += 1
        else:
            self._diff_streak = max(0, self._diff_streak - 1)

        if self._diff_streak >= SCREEN_DIFF_CONFIRM:
            self._diff_streak = 0
            reason = (f"Screen changed: {diff_ratio*100:.0f}% pixels "
                      f"(threshold {SCREEN_DIFF_THRESHOLD*100:.0f}%)")
            return self._save_pair(before, after_full,
                                   reason, "tab_switch_pixel")
        return False

    # ── main loop ────────────────────────────────────────────
    def _run(self):
        while not self._stop_event.is_set():
            # Capture current frame — this is "AFTER" for the current
            # comparison and will become "BEFORE" for the next one.
            current_full  = self._grab_fullres()
            current_thumb = self._thumb_from_fullres(current_full)
            before_full   = self._prev_fullres       # what screen looked like last tick

            # Method A (title) preferred; method B (diff) only if A doesn't fire
            flagged = self._check_title(before_full, current_full)
            if not flagged:
                self._check_pixel_diff(before_full, current_thumb, current_full)

            # Advance rolling state
            self._prev_fullres = current_full

            self._stop_event.wait(SCREEN_SAMPLE_INTERVAL)


# ─────────────────────────────────────────────────────────────
# DETECTION HELPERS
# ─────────────────────────────────────────────────────────────

def _px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def _dist(p1, p2):
    return float(np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float)))

def detect_blink_state(face_lm, w, h):
    def eye_gap(idx):
        return _dist(_px(face_lm.landmark[idx[0]], w, h),
                     _px(face_lm.landmark[idx[1]], w, h))
    avg = (eye_gap(LEFT_BLINK_LM) + eye_gap(RIGHT_BLINK_LM)) / 2.0
    if avg < BLINK_CLOSED_THRESHOLD:  return "closed"
    if avg < PARTIAL_BLINK_THRESHOLD: return "partial"
    return "open"

def detect_iris(eye_lm, face_lm, w, h):
    pts = [face_lm.landmark[i] for i in eye_lm]
    u_y = pts[0].y*h;  l_y = pts[1].y*h
    lx  = pts[2].x*w;  rx  = pts[3].x*w
    cx  = pts[4].x*w;  cy  = pts[4].y*h
    ew  = abs(rx - lx);  eh = abs(l_y - u_y)
    if ew < 1: return "Center"
    h_ratio = max(0.0, min(1.0, (cx - min(lx, rx)) / ew))
    v_ratio = max(0.0, min(1.0, (cy - min(u_y, l_y)) / eh)) if eh > 1 else 0.5
    if h_ratio < 0.35: return "Right"
    if h_ratio > 0.65: return "Left"
    if v_ratio < 0.35: return "Up"
    return "Center"

def get_head_pose(face_lm, w, h):
    img_pts = np.array(
        [(face_lm.landmark[i].x * w, face_lm.landmark[i].y * h) for i in HEAD_LM_IDS],
        dtype=np.float64)
    cam_mat = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(HEAD_3D_POINTS, img_pts, cam_mat,
                                np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    return (float(np.degrees(np.arcsin(-rmat[2, 1]))),
            float(np.degrees(np.arctan2(rmat[2, 0], rmat[2, 2]))),
            float(np.degrees(np.arctan2(rmat[0, 1], rmat[1, 1]))))

def mouth_open_dist(face_lm, w, h):
    return _dist(_px(face_lm.landmark[MOUTH_TOP_LM], w, h),
                 _px(face_lm.landmark[MOUTH_BOTTOM_LM], w, h))

def draw_hud(frame, s: dict):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (480, 340), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    yaw_dir      = ("L " if s["yaw"] < 0 else "R ") + f"{abs(s['yaw']):.0f}deg"
    partial_warn = (f"  ! partial {s['partial_yaw_f']}/{HEAD_YAW_PARTIAL_FRAMES}"
                    if s["partial_yaw_f"] > 0 else "")
    mouth_col    = ((0, 0, 255) if s["mouth_act_pct"] >= MOUTH_MOVEMENT_THRESHOLD * 100
                    else (200, 200, 0))
    tab_col      = (0, 0, 255) if s["tab_switches"] > 0 else (100, 255, 100)
    progress_pct = (s["frame_no"] / s["total_frames"] * 100) if s["total_frames"] > 0 else 0
    up_str       = f"  UP:{s['up_dur']:.1f}s" if s["up_dur"] else ""
    lines = [
        (f"Blinks : {s['blinks']}",                                                    (0, 255, 0)),
        (f"Cheating Count : {s['cheat_count']}",                                       (255, 0, 255)),
        (f"Cheating % : {s['cheat_pct']:.1f}%",                                        (255, 0, 255)),
        (f"Tab Switches : {s['tab_switches']}",                                        tab_col),
        (f"Attention : {s['attention']:.1f}%",                                         (0, 200, 255)),
        (f"Gaze L={s['gaze_l']} R={s['gaze_r']}{up_str}",                             (0, 0, 255)),
        (f"Yaw={yaw_dir}  Pitch={s['pitch']:.0f}  Roll={s['roll']:.0f}{partial_warn}", (0, 165, 255)),
        (f"Mouth gap:{s['mouth_gap']:.1f}px  Active:{s['mouth_act_pct']:.0f}%",        mouth_col),
        (f"Eye state : {s['blink_state']}",                                            (200, 200, 200)),
        (f"Extra body : {'YES' if s['extra_body'] else 'no'}",
         (0, 0, 255) if s["extra_body"] else (100, 255, 100)),
        (f"Frame: {s['frame_no']}/{s['total_frames']} ({progress_pct:.0f}%)",          (180, 180, 180)),
    ]
    for i, (txt, col) in enumerate(lines):
        cv2.putText(frame, txt, (10, 25 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    bar_y = frame.shape[0] - 8
    bar_w = int(frame.shape[1] * progress_pct / 100)
    cv2.rectangle(frame, (0, bar_y), (frame.shape[1], frame.shape[0]), (50, 50, 50), -1)
    cv2.rectangle(frame, (0, bar_y), (bar_w, frame.shape[0]), (0, 200, 100), -1)

# ─────────────────────────────────────────────────────────────
# CORE ANALYSIS
# ─────────────────────────────────────────────────────────────

def run_analysis(job_id: str, video_path: str, exam_window_hint: str = ""):
    job_dir        = RESULTS_DIR / job_id
    screenshot_dir = job_dir / "screenshots"
    job_dir.mkdir(parents=True, exist_ok=True)
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "session_log.json"

    def set_job(**kwargs):
        with jobs_lock:
            jobs[job_id].update(kwargs)

    set_job(status="running")

    screen_monitor = ScreenMonitor(
        job_id           = job_id,
        screenshot_dir   = screenshot_dir,
        event_log        = [],
        exam_window_hint = exam_window_hint,
    )
    if MSS_AVAILABLE:
        screen_monitor.start()
        set_job(screen_monitoring=True)
    else:
        set_job(screen_monitoring=False)

    blink_count             = 0
    eye_closed_frames       = 0
    eye_open                = True
    closed_start_time       = None
    blink_times             = deque()
    gaze_off_frames         = 0
    gaze_up_start_time      = None
    head_pose_frames        = 0
    head_yaw_partial_frames = 0
    mouth_open_frames       = 0
    mouth_partial_frames    = 0
    mouth_activity_log      = deque()
    extra_body_frames       = 0
    bg_motion_streak        = 0
    cheating_count          = 0
    cheating_frames         = 0
    total_frames            = 0
    attention_score         = 100.0
    last_shot_time: dict    = {}
    event_log: list         = []
    gaze_l_str = gaze_r_str = "–"
    pitch_v = yaw_v = roll_v = 0.0
    mouth_gap_v              = 0.0
    blink_state_v            = "–"
    last_body_check_time     = 0.0

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    bg_sub   = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=60, detectShadows=False)
    face_det = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.4)

    def save_shot(frame, reason, reason_key):
        nonlocal cheating_count
        now = time.time()
        if now - last_shot_time.get(reason_key, 0) >= SCREENSHOT_COOLDOWN:
            last_shot_time[reason_key] = now
            cheating_count += 1
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fn = str(screenshot_dir / f"cheat_{reason_key}_{ts}.png")
            cv2.imwrite(fn, frame)
            event_log.append({"time": ts, "reason": reason,
                               "type": "webcam", "file": fn})

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        screen_monitor.stop()
        set_job(status="error", error=f"Cannot open: {video_path}")
        return

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    with mp.solutions.face_mesh.FaceMesh(
            max_num_faces=2, refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            total_frames += 1
            h, w, _ = frame.shape
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            cheating_this_frame = False
            reasons: list = []
            attention_score = min(100.0, attention_score * ATTENTION_DECAY + 0.05)
            fg_mask = bg_sub.apply(frame)

            live_tab_switches = screen_monitor.tab_switch_count if MSS_AVAILABLE else 0

            now_t = time.time()
            if now_t - last_body_check_time >= BODY_CHECK_INTERVAL:
                last_body_check_time = now_t
                mesh_count  = len(result.multi_face_landmarks) if result.multi_face_landmarks else 0
                extra_found = False

                fd_result = face_det.process(rgb)
                fd_count  = len(fd_result.detections) if fd_result.detections else 0
                if fd_result.detections:
                    for det in fd_result.detections:
                        bb = det.location_data.relative_bounding_box
                        x1 = int(bb.xmin * w);  y1 = int(bb.ymin * h)
                        x2 = int((bb.xmin + bb.width) * w)
                        y2 = int((bb.ymin + bb.height) * h)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(frame, "face", (x1, y1-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                if fd_count > max(mesh_count, 1):
                    extra_found = True

                if not extra_found:
                    small = cv2.resize(frame, (320, 240))
                    bodies, _ = hog.detectMultiScale(small, winStride=(6, 6),
                                                     padding=(8, 8), scale=1.03)
                    sx = w / 320; sy = h / 240
                    for (bx, by, bw2, bh2) in bodies:
                        cv2.rectangle(frame,
                                      (int(bx*sx), int(by*sy)),
                                      (int((bx+bw2)*sx), int((by+bh2)*sy)),
                                      (0, 165, 0), 2)
                        cv2.putText(frame, "body", (int(bx*sx), int(by*sy)-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 0), 1)
                    if len(bodies) > max(mesh_count, 1):
                        extra_found = True

                if not extra_found and total_frames > BG_WARMUP_FRAMES:
                    zx1 = int(w * (1 - BG_SUBJECT_ZONE) / 2)
                    zx2 = int(w * (1 + BG_SUBJECT_ZONE) / 2)
                    pmask = fg_mask.copy(); pmask[:, zx1:zx2] = 0
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                    pmask = cv2.morphologyEx(pmask, cv2.MORPH_OPEN,   k)
                    pmask = cv2.morphologyEx(pmask, cv2.MORPH_DILATE, k)
                    contours, _ = cv2.findContours(pmask, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
                    la = 0; lb = None
                    for cnt in contours:
                        a = cv2.contourArea(cnt)
                        if a > la: la = a; lb = cv2.boundingRect(cnt)
                    if la > BG_MIN_AREA and lb and lb[3] > h * BG_MIN_HEIGHT_RATIO:
                        bg_motion_streak += 1
                        x, y, cw, ch = lb
                        cv2.rectangle(frame, (x, y), (x+cw, y+ch), (0, 0, 255), 2)
                        cv2.putText(frame, "PERSON?", (x, y-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        if bg_motion_streak >= BG_MOTION_CONFIRM:
                            extra_found = True
                    else:
                        bg_motion_streak = max(0, bg_motion_streak - 1)

                if extra_found: extra_body_frames += 1
                else:           extra_body_frames = max(0, extra_body_frames - 1)

            if extra_body_frames >= EXTRA_BODY_FRAMES:
                cheating_this_frame = True
                reasons.append(("Extra person detected in frame!", "extra_body"))

            if result.multi_face_landmarks:
                if len(result.multi_face_landmarks) > 1:
                    cheating_this_frame = True
                    reasons.append((f"{len(result.multi_face_landmarks)} faces detected!", "multi_face"))
                    for idx, efl in enumerate(result.multi_face_landmarks[1:], start=2):
                        xs = [int(lm.x * w) for lm in efl.landmark]
                        ys = [int(lm.y * h) for lm in efl.landmark]
                        cv2.rectangle(frame, (min(xs), min(ys)),
                                      (max(xs), max(ys)), (0, 0, 255), 3)
                        cv2.putText(frame, f"EXTRA PERSON #{idx}",
                                    (min(xs), min(ys)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                fl = result.multi_face_landmarks[0]

                blink_state_v = detect_blink_state(fl, w, h)
                if blink_state_v == "closed":
                    eye_closed_frames += 1
                    if eye_closed_frames >= MAX_CLOSED_FRAMES and eye_open:
                        blink_count += 1; eye_open = False
                        blink_times.append(time.time())
                    if closed_start_time is None:
                        closed_start_time = time.time()
                    else:
                        dur = time.time() - closed_start_time
                        if dur > MAX_EYE_CLOSED_DURATION:
                            cheating_this_frame = True
                            reasons.append((f"Eyes closed {dur:.1f}s", "long_blink"))
                elif blink_state_v == "partial":
                    eye_closed_frames = 0; eye_open = True; closed_start_time = None
                else:
                    eye_closed_frames = 0; eye_open = True; closed_start_time = None

                now = time.time()
                while blink_times and now - blink_times[0] > RAPID_BLINK_WINDOW:
                    blink_times.popleft()
                if len(blink_times) >= RAPID_BLINK_COUNT:
                    cheating_this_frame = True
                    reasons.append((f"Rapid blinking ({len(blink_times)} in "
                                    f"{RAPID_BLINK_WINDOW:.0f}s)", "rapid_blink"))

                if blink_state_v == "open":
                    gaze_l_str = detect_iris(LEFT_EYE_LM,  fl, w, h)
                    gaze_r_str = detect_iris(RIGHT_EYE_LM, fl, w, h)
                    sideways = (gaze_l_str in ["Left", "Right"] or
                                gaze_r_str in ["Left", "Right"])
                    if sideways:
                        gaze_off_frames += 1; gaze_up_start_time = None
                    else:
                        gaze_off_frames = 0
                    if gaze_off_frames >= GAZE_SUSTAINED_FRAMES:
                        cheating_this_frame = True
                        reasons.append((f"Sideways gaze L={gaze_l_str} R={gaze_r_str}",
                                        "gaze"))
                    both_up = (gaze_l_str == "Up" and gaze_r_str == "Up")
                    if both_up:
                        if gaze_up_start_time is None:
                            gaze_up_start_time = time.time()
                        else:
                            up_dur = time.time() - gaze_up_start_time
                            if up_dur >= GAZE_UP_MAX_DURATION:
                                cheating_this_frame = True
                                reasons.append((f"Looking up {up_dur:.1f}s (reading notes?)",
                                                "gaze_up"))
                    else:
                        gaze_up_start_time = None
                else:
                    gaze_l_str = gaze_r_str = "–"
                    gaze_off_frames = 0; gaze_up_start_time = None

                pitch_v, yaw_v, roll_v = get_head_pose(fl, w, h)
                head_deviated = (abs(pitch_v) > HEAD_PITCH_LIMIT or
                                 abs(yaw_v)   > HEAD_YAW_LIMIT   or
                                 abs(roll_v)  > HEAD_ROLL_LIMIT)
                head_pose_frames = head_pose_frames + 1 if head_deviated else 0
                if head_pose_frames >= HEAD_POSE_FRAMES:
                    cheating_this_frame = True
                    d = "Left" if yaw_v < 0 else "Right"
                    reasons.append((f"Head turned {d} P={pitch_v:.0f} "
                                    f"Y={yaw_v:.0f} R={roll_v:.0f}", "head_pose"))

                partial_yaw = (HEAD_YAW_PARTIAL_LIMIT < abs(yaw_v) <= HEAD_YAW_LIMIT)
                head_yaw_partial_frames = (head_yaw_partial_frames + 1 if partial_yaw
                                           else max(0, head_yaw_partial_frames - 1))
                if head_yaw_partial_frames >= HEAD_YAW_PARTIAL_FRAMES:
                    cheating_this_frame = True
                    d = "Left" if yaw_v < 0 else "Right"
                    reasons.append((f"Partial head glance {d} ({yaw_v:.0f}deg)",
                                    "head_partial"))

                mouth_gap_v = mouth_open_dist(fl, w, h)
                now_m = time.time()
                if mouth_gap_v > MOUTH_PARTIAL_THRESHOLD:
                    mouth_activity_log.append(now_m)
                while mouth_activity_log and now_m - mouth_activity_log[0] > MOUTH_MOVEMENT_WINDOW:
                    mouth_activity_log.popleft()

                if mouth_gap_v > MOUTH_OPEN_THRESHOLD:
                    mouth_open_frames += 1; mouth_partial_frames = 0
                elif mouth_gap_v > MOUTH_PARTIAL_THRESHOLD:
                    mouth_partial_frames += 1
                    mouth_open_frames = max(0, mouth_open_frames - 1)
                else:
                    mouth_open_frames    = max(0, mouth_open_frames    - 1)
                    mouth_partial_frames = max(0, mouth_partial_frames - 1)

                if mouth_open_frames >= MOUTH_OPEN_FRAMES:
                    cheating_this_frame = True
                    reasons.append((f"Mouth open whispering ({mouth_gap_v:.1f}px)", "mouth"))
                elif mouth_partial_frames >= MOUTH_PARTIAL_FRAMES:
                    cheating_this_frame = True
                    reasons.append((f"Mouth partial open mumbling ({mouth_gap_v:.1f}px)",
                                    "mouth_partial"))
                else:
                    act_ratio = len(mouth_activity_log) / (MOUTH_MOVEMENT_WINDOW * 30)
                    if act_ratio >= MOUTH_MOVEMENT_THRESHOLD:
                        cheating_this_frame = True
                        reasons.append((f"Continuous mouth movement "
                                        f"({act_ratio*100:.0f}% active)",
                                        "mouth_continuous"))
            else:
                gaze_l_str = gaze_r_str = "–"; blink_state_v = "no face"
                head_pose_frames = mouth_open_frames = mouth_partial_frames = 0
                mouth_activity_log.clear(); gaze_up_start_time = None
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if cv2.countNonZero(gray) > 0.2 * w * h:
                    cheating_this_frame = True
                    reasons.append(("Face not visible", "no_face"))

            if cheating_this_frame:
                cheating_frames += 1
                attention_score  = max(0.0, attention_score - ATTENTION_PENALTY)
                cv2.rectangle(frame, (0, h-50), (w, h), (0, 0, 200), -1)
                short = " | ".join(r[0] for r in reasons)
                cv2.putText(frame, f"ALERT: {short[:80]}", (10, h-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cheat_pct     = (cheating_frames / total_frames * 100) if total_frames else 0
            mouth_act_pct = min(100.0,
                (len(mouth_activity_log) / (MOUTH_MOVEMENT_WINDOW * 30)) * 100)
            up_dur = (time.time() - gaze_up_start_time) if gaze_up_start_time else 0.0

            draw_hud(frame, dict(
                blinks=blink_count, cheat_count=cheating_count, cheat_pct=cheat_pct,
                attention=attention_score, gaze_l=gaze_l_str, gaze_r=gaze_r_str,
                pitch=pitch_v, yaw=yaw_v, roll=roll_v, mouth_gap=mouth_gap_v,
                mouth_act_pct=mouth_act_pct, blink_state=blink_state_v,
                extra_body=extra_body_frames >= EXTRA_BODY_FRAMES,
                partial_yaw_f=head_yaw_partial_frames,
                frame_no=total_frames, total_frames=total_video_frames, up_dur=up_dur,
                tab_switches=live_tab_switches,
            ))

            if cheating_this_frame:
                for reason_text, reason_key in reasons:
                    save_shot(frame, reason_text, reason_key)

            if total_frames % 30 == 0:
                set_job(
                    progress         = round((total_frames / total_video_frames) * 100, 1),
                    frames_done      = total_frames,
                    cheat_count      = cheating_count + screen_monitor.tab_switch_count,
                    cheat_pct        = round(cheat_pct, 2),
                    blinks           = blink_count,
                    attention_score  = round(attention_score, 1),
                    tab_switch_count = screen_monitor.tab_switch_count,
                )

    cap.release()
    screen_monitor.stop()

    combined_events = event_log + screen_monitor.event_log
    combined_events.sort(key=lambda e: e["time"])
    total_cheat_count = cheating_count + screen_monitor.tab_switch_count
    cheat_pct = (cheating_frames / total_frames * 100) if total_frames else 0

    summary = {
        "job_id"            : job_id,
        "total_frames"      : total_frames,
        "blinks"            : blink_count,
        "cheating_count"    : total_cheat_count,
        "webcam_cheat_count": cheating_count,
        "tab_switch_count"  : screen_monitor.tab_switch_count,
        "cheating_pct"      : round(cheat_pct, 2),
        "attention_score"   : round(attention_score, 1),
        "screen_monitoring" : MSS_AVAILABLE,
        "events"            : combined_events,
    }
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)

    set_job(
        status           = "done",
        progress         = 100.0,
        frames_done      = total_frames,
        cheat_count      = total_cheat_count,
        cheat_pct        = round(cheat_pct, 2),
        blinks           = blink_count,
        attention_score  = round(attention_score, 1),
        tab_switch_count = screen_monitor.tab_switch_count,
        log_path         = str(log_path),
        screenshot_dir   = str(screenshot_dir),
        events           = combined_events,
    )

# ─────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Cheating Detection API",
    description=(
        "Upload exam videos and detect cheating behaviour.\n\n"
        "On every confirmed tab/window switch **two full-resolution screenshots** "
        "are saved automatically:\n"
        "- `*_BEFORE.png` — what the screen showed just before the switch\n"
        "- `*_AFTER.png`  — what the screen showed just after the switch\n\n"
        "Both carry a red alert banner with the reason and timestamp."
    ),
    version="3.0.0",
)

@app.post("/analyze", summary="Upload a video and start analysis")
async def analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file (.mp4 .avi .mkv .mov)"),
    exam_window_hint: str = "",
):
    ext = Path(file.filename).suffix.lower()
    if ext not in {".mp4", ".avi", ".mkv", ".mov", ".wmv"}:
        raise HTTPException(400, f"Unsupported file type '{ext}'.")

    job_id    = str(uuid.uuid4())
    save_path = str(UPLOAD_DIR / f"{job_id}{ext}")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    with jobs_lock:
        jobs[job_id] = {
            "status"           : "queued",
            "filename"         : file.filename,
            "progress"         : 0.0,
            "frames_done"      : 0,
            "cheat_count"      : 0,
            "cheat_pct"        : 0.0,
            "blinks"           : 0,
            "attention_score"  : 100.0,
            "tab_switch_count" : 0,
            "screen_monitoring": MSS_AVAILABLE,
            "events"           : [],
        }

    background_tasks.add_task(run_analysis, job_id, save_path, exam_window_hint)
    return {
        "job_id"           : job_id,
        "screen_monitoring": MSS_AVAILABLE,
        "title_tracking"   : PYGETWINDOW_AVAILABLE,
        "message"          : f"Analysis started. Poll /status/{job_id} for live updates.",
    }

@app.get("/status/{job_id}")
async def status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    return job

@app.get("/results/{job_id}/log")
async def download_log(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Results not ready or job not found.")
    log_path = job.get("log_path", "")
    if not Path(log_path).exists():
        raise HTTPException(404, "Log file not found on disk.")
    return FileResponse(log_path, media_type="application/json",
                        filename=f"session_log_{job_id}.json")

@app.get("/results/{job_id}/screenshots")
async def list_screenshots(job_id: str):
    """
    Returns screenshot filenames grouped into three buckets:
    - webcam_screenshots      — face/behaviour detections from the video
    - tab_switch_before_shots — full-res screen captured BEFORE each switch
    - tab_switch_after_shots  — full-res screen captured AFTER each switch
    """
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Results not ready or job not found.")
    shot_dir = Path(job.get("screenshot_dir", ""))
    files    = sorted([f.name for f in shot_dir.glob("*.png")]) if shot_dir.exists() else []

    before = [f for f in files if "_BEFORE.png" in f]
    after  = [f for f in files if "_AFTER.png"  in f]
    webcam = [f for f in files if "tab_switch"  not in f]

    return {
        "job_id"                 : job_id,
        "total_count"            : len(files),
        "webcam_count"           : len(webcam),
        "tab_switch_before_count": len(before),
        "tab_switch_after_count" : len(after),
        "webcam_screenshots"     : webcam,
        "tab_switch_before_shots": before,
        "tab_switch_after_shots" : after,
    }

@app.get("/results/{job_id}/screenshots/{filename}")
async def get_screenshot(job_id: str, filename: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    shot_path = RESULTS_DIR / job_id / "screenshots" / filename
    if not shot_path.exists():
        raise HTTPException(404, f"Screenshot '{filename}' not found.")
    return FileResponse(str(shot_path), media_type="image/png")

@app.get("/jobs")
async def list_jobs():
    with jobs_lock:
        return [{"job_id": jid, "filename": j.get("filename"),
                 "status": j.get("status"), "progress": j.get("progress"),
                 "cheat_pct": j.get("cheat_pct"),
                 "tab_switch_count": j.get("tab_switch_count", 0)}
                for jid, j in jobs.items()]

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(404, "Job not found.")
        jobs.pop(job_id)
    job_dir = RESULTS_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    for ext in [".mp4", ".avi", ".mkv", ".mov", ".wmv"]:
        p = UPLOAD_DIR / f"{job_id}{ext}"
        if p.exists(): p.unlink()
    return {"message": f"Job '{job_id}' deleted successfully."}

@app.get("/health")
async def health():
    with jobs_lock:
        active = sum(1 for j in jobs.values() if j["status"] == "running")
    return {"status": "ok", "active_jobs": active, "total_jobs": len(jobs),
            "screen_monitoring": MSS_AVAILABLE, "title_tracking": PYGETWINDOW_AVAILABLE}

if __name__ == "__main__":
    uvicorn.run("cheating_api:app", host="127.0.0.1", port=8000, reload=True)