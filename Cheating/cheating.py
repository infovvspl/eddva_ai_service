import cv2
import mediapipe as mp
import numpy as np
import datetime
import time
import os
import sys
import json
from collections import deque

# Suppress TensorFlow and system warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '3'          # suppresses MediaPipe/glog INFO lines
stderr_backup = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stdout_backup = sys.stdout                 # backup so session report still prints
try:
    from absl import logging
    logging.set_verbosity(logging.ERROR)
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
screenshot_dir = r"D:\VVSPL\AI Study Planner\Cheating\cheating_screenshots"
log_path       = os.path.join(screenshot_dir, "session_log.json")
os.makedirs(screenshot_dir, exist_ok=True)

# ── Blink thresholds ──────────────────────────────────────────
BLINK_CLOSED_THRESHOLD   = 4.0   # px – eye is fully closed
PARTIAL_BLINK_THRESHOLD  = 5.5   # px – suspicious partial closure
MAX_CLOSED_FRAMES        = 2     # frames before counting a blink
MAX_EYE_CLOSED_DURATION  = 3.0   # seconds – long eye-close = suspicious

# ── Iris / gaze ───────────────────────────────────────────────
# Thresholds are now ratio-based (0.0–1.0) inside detect_iris()
GAZE_SUSTAINED_FRAMES    = 6     # consecutive frames of sideways gaze → flag

# ── Head pose ─────────────────────────────────────────────────
HEAD_PITCH_LIMIT         = 25    # degrees – nodding down (reading phone etc.)
HEAD_YAW_LIMIT           = 30    # degrees – full turn left/right
HEAD_ROLL_LIMIT          = 20    # degrees – tilting head
HEAD_POSE_FRAMES         = 8     # sustained frames before flagging full turn

# Partial / quick sideways glance (lower threshold, fewer frames needed)
HEAD_YAW_PARTIAL_LIMIT   = 15    # degrees – subtle sideways glance
HEAD_YAW_PARTIAL_FRAMES  = 12    # must sustain partial turn for this many frames
                                  # (longer window catches intentional glances, ignores fidgeting)

# ── Mouth / lip ───────────────────────────────────────────────
MOUTH_OPEN_THRESHOLD     = 10.0  # px – clearly open mouth (talking/whispering)
MOUTH_PARTIAL_THRESHOLD  = 5.0   # px – partially open (mumbling/mouthing words)
MOUTH_OPEN_FRAMES        = 12    # frames sustained for full open (~0.4s at 30fps)
MOUTH_PARTIAL_FRAMES     = 20    # frames sustained for partial open (~0.7s)
MOUTH_MOVEMENT_WINDOW    = 5.0   # seconds – rolling window to track mouth activity
MOUTH_MOVEMENT_THRESHOLD = 0.40  # fraction of window frames with open mouth = talking

# ── Rapid blink (stress signal) ───────────────────────────────
RAPID_BLINK_WINDOW       = 10.0  # seconds
RAPID_BLINK_COUNT        = 8     # blinks within window → flag

# ── Screenshot / cooldown ─────────────────────────────────────
SCREENSHOT_COOLDOWN      = 2.0   # seconds between saves per reason category

# ── Attention score decay ─────────────────────────────────────
ATTENTION_DECAY          = 0.995  # multiplied every frame; cheating events subtract
ATTENTION_PENALTY        = 5      # subtracted per cheating event (0-100 scale)

# ─────────────────────────────────────────────────────────────
# MEDIAPIPE LANDMARKS
# ─────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh

# ── HOG person detector (upright bodies) ──
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ── MediaPipe Face Detection (catches distant/partial faces) ──
mp_face_detection = mp.solutions.face_detection
face_detector     = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.4
)

# ── MOG2 Background Subtractor ────────────────────────────────
bg_subtractor        = cv2.createBackgroundSubtractorMOG2(
    history=500,          # longer history = much stabler background model
    varThreshold=60,      # higher = less sensitive, ignores lighting flicker/shadows
    detectShadows=False
)
BG_MIN_AREA          = 6000   # px² — must be large enough to be a human torso
BG_MIN_HEIGHT_RATIO  = 0.20   # blob must be at least 20% of frame height
BG_SUBJECT_ZONE      = 0.50   # centre 50% = main subject safe zone
BG_WARMUP_FRAMES     = 120    # wait ~4s for background model to fully stabilise
BG_MOTION_CONFIRM    = 4      # blob must appear in this many consecutive checks before flagging

last_body_check_time = 0
BODY_CHECK_INTERVAL  = 0.2
extra_body_detected  = False

# Iris: [Upper, Lower, Left, Right, Centre]
LEFT_EYE_LM  = [159, 145, 33, 133, 468]
RIGHT_EYE_LM = [386, 374, 263, 362, 473]

# Blink (EAR-based)
LEFT_BLINK_LM  = [159, 145]
RIGHT_BLINK_LM = [386, 374]

# Mouth landmarks (vertical pair)
MOUTH_TOP_LM    = 13
MOUTH_BOTTOM_LM = 14

# Head pose canonical 3-D reference points (in mm)
HEAD_3D_POINTS = np.array([
    (0.0,   0.0,    0.0),    # Nose tip      – landmark 1
    (0.0,  -330.0, -65.0),   # Chin          – landmark 152
    (-225.0, 170.0, -135.0), # Left eye      – landmark 263
    (225.0,  170.0, -135.0), # Right eye     – landmark 33
    (-150.0,-150.0, -125.0), # Left mouth    – landmark 287
    (150.0, -150.0, -125.0), # Right mouth   – landmark 57
], dtype=np.float64)

HEAD_LM_IDS = [1, 152, 263, 33, 287, 57]

# ─────────────────────────────────────────────────────────────
# STATE VARIABLES
# ─────────────────────────────────────────────────────────────
blink_count        = 0
eye_closed_frames  = 0
eye_open           = True
closed_start_time  = None
blink_times        = deque()          # timestamps of recent blinks

gaze_off_frames         = 0
head_pose_frames        = 0
head_yaw_partial_frames = 0
mouth_open_frames       = 0
mouth_partial_frames    = 0      # tracks sustained partial mouth opening
mouth_activity_log      = deque()  # timestamps when mouth was open/partial — rolling window
extra_body_frames    = 0
EXTRA_BODY_FRAMES    = 4      # HOG/face hits needed before flagging
bg_motion_streak     = 0      # consecutive MOG2 checks with large blob

cheating_count     = 0
cheating_frames    = 0
total_frames       = 0
attention_score    = 100.0
last_shot_time     = {}               # reason → last screenshot time
event_log          = []               # list of dicts for JSON report

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def px(lm, w, h):
    """Landmark → pixel coords."""
    return int(lm.x * w), int(lm.y * h)


def dist(p1, p2):
    return np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float))


def detect_blink_state(face_lm, w, h):
    """Return 'closed' / 'partial' / 'open'."""
    def eye_gap(indices):
        u = face_lm.landmark[indices[0]]
        l = face_lm.landmark[indices[1]]
        return dist(px(u, w, h), px(l, w, h))

    avg = (eye_gap(LEFT_BLINK_LM) + eye_gap(RIGHT_BLINK_LM)) / 2.0
    if avg < BLINK_CLOSED_THRESHOLD:
        return "closed"
    elif avg < PARTIAL_BLINK_THRESHOLD:
        return "partial"
    return "open"


def detect_iris(eye_lm, face_lm, w, h):
    """Return gaze direction using normalized iris ratio (works at any distance)."""
    pts  = [face_lm.landmark[i] for i in eye_lm]
    lx   = pts[2].x * w;  rx  = pts[3].x * w
    cx   = pts[4].x * w

    eye_width = abs(rx - lx)
    if eye_width < 1:
        return "Center"

    left_anchor = min(lx, rx)
    h_ratio = (cx - left_anchor) / eye_width
    h_ratio = max(0.0, min(1.0, h_ratio))

    H_LEFT_THRESH  = 0.35
    H_RIGHT_THRESH = 0.65

    if h_ratio < H_LEFT_THRESH:
        return "Right"
    elif h_ratio > H_RIGHT_THRESH:
        return "Left"
    return "Center"


def get_head_pose(face_lm, w, h):
    """Return (pitch, yaw, roll) in degrees using solvePnP."""
    img_pts = np.array(
        [(face_lm.landmark[i].x * w, face_lm.landmark[i].y * h) for i in HEAD_LM_IDS],
        dtype=np.float64
    )
    focal   = w
    cx_img, cy_img = w / 2, h / 2
    cam_mat = np.array([[focal, 0, cx_img], [0, focal, cy_img], [0, 0, 1]], dtype=np.float64)
    dist_co = np.zeros((4, 1))
    ok, rvec, _ = cv2.solvePnP(HEAD_3D_POINTS, img_pts, cam_mat, dist_co,
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0, 0, 0
    rmat, _ = cv2.Rodrigues(rvec)
    # Decompose to Euler angles
    pitch = np.degrees(np.arcsin(-rmat[2, 1]))
    yaw   = np.degrees(np.arctan2(rmat[2, 0], rmat[2, 2]))
    roll  = np.degrees(np.arctan2(rmat[0, 1], rmat[1, 1]))
    return pitch, yaw, roll


def mouth_open_dist(face_lm, w, h):
    """Vertical mouth gap in pixels."""
    top = px(face_lm.landmark[MOUTH_TOP_LM], w, h)
    bot = px(face_lm.landmark[MOUTH_BOTTOM_LM], w, h)
    return dist(top, bot)


def should_screenshot(reason_key):
    """Rate-limit screenshots per category."""
    global last_shot_time
    now = time.time()
    if now - last_shot_time.get(reason_key, 0) >= SCREENSHOT_COOLDOWN:
        last_shot_time[reason_key] = now
        return True
    return False


def save_screenshot(frame, reason, reason_key):
    """Save the fully-annotated frame and log to JSON only — no terminal output."""
    global cheating_count
    if should_screenshot(reason_key):
        cheating_count += 1
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = os.path.join(screenshot_dir, f"cheat_{reason_key}_{ts}.png")
        cv2.imwrite(fn, frame)
        event_log.append({"time": ts, "reason": reason, "file": fn})


def draw_hud(frame, blinks, cheat_count, cheat_pct, attention,
             pitch, yaw, roll, gaze_l, gaze_r, mouth_gap, blink_state, extra_body,
             partial_yaw_frames, mouth_activity_pct):
    """Overlay all stats onto the frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (390, 330), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    yaw_dir      = ("L " if yaw < 0 else "R ") + f"{abs(yaw):.0f}deg"
    partial_warn = f"  ! partial {partial_yaw_frames}/{HEAD_YAW_PARTIAL_FRAMES}" if partial_yaw_frames > 0 else ""
    mouth_col    = (0, 0, 255) if mouth_activity_pct >= MOUTH_MOVEMENT_THRESHOLD * 100 else (200, 200, 0)

    lines = [
        (f"Blinks : {blinks}",                                             (0, 255, 0)),
        (f"Cheating Count : {cheat_count}",                                (255, 0, 255)),
        (f"Cheating % : {cheat_pct:.1f}%",                                 (255, 0, 255)),
        (f"Attention : {attention:.1f}%",                                  (0, 200, 255)),
        (f"Gaze  L={gaze_l}  R={gaze_r}",                                 (0, 0, 255)),
        (f"Yaw={yaw_dir}  Pitch={pitch:.0f}  Roll={roll:.0f}{partial_warn}", (0, 165, 255)),
        (f"Mouth gap:{mouth_gap:.1f}px  Active:{mouth_activity_pct:.0f}%", mouth_col),
        (f"Eye state : {blink_state}",                                     (200, 200, 200)),
        (f"Extra body : {'YES' if extra_body else 'no'}",                  (0, 0, 255) if extra_body else (100, 255, 100)),
    ]
    for i, (txt, col) in enumerate(lines):
        cv2.putText(frame, txt, (10, 25 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cv2.namedWindow('Cheating & Blink Detection', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Cheating & Blink Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Defaults shown in HUD even before a face is found
gaze_l_str = gaze_r_str = "–"
pitch_v = yaw_v = roll_v = 0.0
mouth_gap_v = 0.0
blink_state_v = "–"
cheating_pct  = 0.0

with mp_face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        total_frames += 1
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        h, w, _ = frame.shape

        cheating_this_frame = False
        reasons = []

        # ── Attention score drift ────────────────────────────
        attention_score = min(100.0, attention_score * ATTENTION_DECAY + 0.05)

        # ── Person detection (3 methods) ────────────────────
        # Always feed MOG2 every frame so its model stays current
        fg_mask = bg_subtractor.apply(frame)

        now_t = time.time()
        if now_t - last_body_check_time >= BODY_CHECK_INTERVAL:
            last_body_check_time = now_t
            mesh_count = len(result.multi_face_landmarks) if result.multi_face_landmarks else 0
            extra_found = False

            # ── Method 1: MediaPipe face detector ───────────
            # Catches distant/side faces that face_mesh misses
            fd_result = face_detector.process(rgb)
            fd_count  = len(fd_result.detections) if fd_result.detections else 0
            if fd_result.detections:
                for det in fd_result.detections:
                    bb = det.location_data.relative_bounding_box
                    x1 = int(bb.xmin * w);           y1 = int(bb.ymin * h)
                    x2 = int((bb.xmin + bb.width) * w)
                    y2 = int((bb.ymin + bb.height) * h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, "face", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if fd_count > max(mesh_count, 1):
                extra_found = True

            # ── Method 2: HOG body detector ─────────────────
            # Catches upright people, frontal or side view
            if not extra_found:
                small     = cv2.resize(frame, (320, 240))
                bodies, _ = hog.detectMultiScale(small, winStride=(6, 6),
                                                  padding=(8, 8), scale=1.03)
                sx = w / 320;  sy = h / 240
                for (bx, by, bw2, bh2) in bodies:
                    x1, y1 = int(bx * sx), int(by * sy)
                    x2, y2 = int((bx + bw2) * sx), int((by + bh2) * sy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 0), 2)
                    cv2.putText(frame, "body", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 0), 1)
                if len(bodies) > max(mesh_count, 1):
                    extra_found = True

            # ── Method 3: Background subtraction (MOG2) ────────
            # Catches backs, arms, partial bodies — anything moving
            # outside the main subject's central zone
            if not extra_found and total_frames > BG_WARMUP_FRAMES:
                zone_x1 = int(w * (1 - BG_SUBJECT_ZONE) / 2)
                zone_x2 = int(w * (1 + BG_SUBJECT_ZONE) / 2)

                periphery_mask = fg_mask.copy()
                periphery_mask[:, zone_x1:zone_x2] = 0

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                periphery_mask = cv2.morphologyEx(periphery_mask, cv2.MORPH_OPEN,   kernel)
                periphery_mask = cv2.morphologyEx(periphery_mask, cv2.MORPH_DILATE, kernel)

                contours, _ = cv2.findContours(periphery_mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

                largest_area = 0
                largest_box  = None
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > largest_area:
                        largest_area = area
                        largest_box  = cv2.boundingRect(cnt)

                # Blob must be large AND tall enough to be a human body
                if (largest_area > BG_MIN_AREA and largest_box is not None
                        and largest_box[3] > h * BG_MIN_HEIGHT_RATIO):
                    bg_motion_streak += 1
                    x, y, cw, ch = largest_box
                    cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 0, 255), 2)
                    cv2.putText(frame, "PERSON?", (x, y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    # Only flag after BG_MOTION_CONFIRM consecutive detections
                    if bg_motion_streak >= BG_MOTION_CONFIRM:
                        extra_found = True
                else:
                    bg_motion_streak = max(0, bg_motion_streak - 1)  # decay gradually

            if extra_found:
                extra_body_frames += 1
            else:
                extra_body_frames = max(0, extra_body_frames - 1)

        if extra_body_frames >= EXTRA_BODY_FRAMES:
            cheating_this_frame = True
            reasons.append(("Extra person detected in frame!", "extra_body"))

        # ── Face presence check ──────────────────────────────
        if result.multi_face_landmarks:

            # ── Multiple faces ───────────────────────────────
            if len(result.multi_face_landmarks) > 1:
                cheating_this_frame = True
                reasons.append((f"{len(result.multi_face_landmarks)} faces detected!", "multi_face"))
                # Draw red box around every extra face
                for idx, extra_fl in enumerate(result.multi_face_landmarks[1:], start=2):
                    xs = [int(lm.x * w) for lm in extra_fl.landmark]
                    ys = [int(lm.y * h) for lm in extra_fl.landmark]
                    cv2.rectangle(frame, (min(xs), min(ys)), (max(xs), max(ys)), (0, 0, 255), 3)
                    cv2.putText(frame, f"EXTRA PERSON #{idx}", (min(xs), min(ys) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            fl = result.multi_face_landmarks[0]

            # ── Blink / eye state ────────────────────────────
            blink_state_v = detect_blink_state(fl, w, h)

            if blink_state_v == "closed":
                eye_closed_frames += 1
                if eye_closed_frames >= MAX_CLOSED_FRAMES and eye_open:
                    blink_count += 1
                    eye_open = False
                    blink_times.append(time.time())

                if closed_start_time is None:
                    closed_start_time = time.time()
                else:
                    dur = time.time() - closed_start_time
                    if dur > MAX_EYE_CLOSED_DURATION:
                        cheating_this_frame = True
                        reasons.append((f"Eyes closed {dur:.1f}s", "long_blink"))

            elif blink_state_v == "partial":
                # Partial closure – transition state during normal blink, do NOT flag
                # Only reset timers, treat same as open
                eye_closed_frames = 0
                eye_open          = True
                closed_start_time = None

            else:  # open
                eye_closed_frames = 0
                eye_open          = True
                closed_start_time = None

            # Prune old blink timestamps
            now = time.time()
            while blink_times and now - blink_times[0] > RAPID_BLINK_WINDOW:
                blink_times.popleft()
            if len(blink_times) >= RAPID_BLINK_COUNT:
                cheating_this_frame = True
                reasons.append((f"Rapid blinking ({len(blink_times)} in {RAPID_BLINK_WINDOW:.0f}s)",
                                 "rapid_blink"))

            # ── Iris / gaze tracking ─────────────────────────
            if blink_state_v == "open":
                gaze_l_str = detect_iris(LEFT_EYE_LM,  fl, w, h)
                gaze_r_str = detect_iris(RIGHT_EYE_LM, fl, w, h)

                # Only sideways gaze is suspicious (copying from neighbour).
                # Up = thinking, Down = reading own paper — both are normal.
                sideways_l = gaze_l_str in ["Left", "Right"]
                sideways_r = gaze_r_str in ["Left", "Right"]

                if sideways_l or sideways_r:
                    gaze_off_frames += 1
                else:
                    gaze_off_frames = 0

                if gaze_off_frames >= GAZE_SUSTAINED_FRAMES:
                    cheating_this_frame = True
                    reasons.append((f"Sideways gaze L={gaze_l_str} R={gaze_r_str}",
                                     "gaze"))
            else:
                gaze_l_str = gaze_r_str = "–"
                gaze_off_frames = 0

            # ── Head pose ────────────────────────────────────
            pitch_v, yaw_v, roll_v = get_head_pose(fl, w, h)

            # Full turn (large angle, fewer frames needed)
            head_deviated = (abs(pitch_v) > HEAD_PITCH_LIMIT or
                             abs(yaw_v)   > HEAD_YAW_LIMIT   or
                             abs(roll_v)  > HEAD_ROLL_LIMIT)
            if head_deviated:
                head_pose_frames += 1
            else:
                head_pose_frames = 0

            if head_pose_frames >= HEAD_POSE_FRAMES:
                cheating_this_frame = True
                dir_str = "Left" if yaw_v < 0 else "Right"
                reasons.append((f"Head turned {dir_str} P={pitch_v:.0f} Y={yaw_v:.0f} R={roll_v:.0f}",
                                 "head_pose"))

            # Partial / subtle sideways glance (lower angle, more frames needed)
            # Only yaw (left/right) — pitch/roll alone are not glancing
            partial_yaw = (abs(yaw_v) > HEAD_YAW_PARTIAL_LIMIT and
                           abs(yaw_v) <= HEAD_YAW_LIMIT)           # between partial and full
            if partial_yaw:
                head_yaw_partial_frames += 1
            else:
                head_yaw_partial_frames = max(0, head_yaw_partial_frames - 1)  # decay gradually

            if head_yaw_partial_frames >= HEAD_YAW_PARTIAL_FRAMES:
                cheating_this_frame = True
                dir_str = "Left" if yaw_v < 0 else "Right"
                reasons.append((f"Partial head glance {dir_str} ({yaw_v:.0f}deg)",
                                 "head_partial"))

            # ── Mouth / whispering / continuous talking ───────
            mouth_gap_v = mouth_open_dist(fl, w, h)
            now_m       = time.time()

            # Log every frame where mouth is open or partially open
            if mouth_gap_v > MOUTH_PARTIAL_THRESHOLD:
                mouth_activity_log.append(now_m)

            # Prune entries outside the rolling window
            while mouth_activity_log and now_m - mouth_activity_log[0] > MOUTH_MOVEMENT_WINDOW:
                mouth_activity_log.popleft()

            # ── Tier 1: Sustained fully open mouth ──────────
            if mouth_gap_v > MOUTH_OPEN_THRESHOLD:
                mouth_open_frames    += 1
                mouth_partial_frames  = 0
            elif mouth_gap_v > MOUTH_PARTIAL_THRESHOLD:
                mouth_partial_frames += 1
                mouth_open_frames     = max(0, mouth_open_frames - 1)
            else:
                mouth_open_frames    = max(0, mouth_open_frames    - 1)
                mouth_partial_frames = max(0, mouth_partial_frames - 1)

            if mouth_open_frames >= MOUTH_OPEN_FRAMES:
                cheating_this_frame = True
                reasons.append((f"Mouth open whispering ({mouth_gap_v:.1f}px)", "mouth"))

            # ── Tier 2: Sustained partial open mouth ────────
            elif mouth_partial_frames >= MOUTH_PARTIAL_FRAMES:
                cheating_this_frame = True
                reasons.append((f"Mouth partial open mumbling ({mouth_gap_v:.1f}px)", "mouth_partial"))

            # ── Tier 3: Continuous/repeated mouth movement ──
            # If mouth was open for >40% of the last 5 seconds = talking continuously
            else:
                expected_frames = MOUTH_MOVEMENT_WINDOW * 30  # assume ~30fps
                activity_ratio  = len(mouth_activity_log) / expected_frames
                if activity_ratio >= MOUTH_MOVEMENT_THRESHOLD:
                    cheating_this_frame = True
                    reasons.append((f"Continuous mouth movement ({activity_ratio*100:.0f}% active)",
                                    "mouth_continuous"))

        else:
            # ── No face detected ─────────────────────────────
            gaze_l_str = gaze_r_str = "–"
            blink_state_v          = "no face"
            head_pose_frames       = 0
            mouth_open_frames      = 0
            mouth_partial_frames   = 0
            mouth_activity_log.clear()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if cv2.countNonZero(gray) > 0.2 * w * h:
                cheating_this_frame = True
                reasons.append(("Face not visible", "no_face"))

        # ── Process flags ────────────────────────────────────
        if cheating_this_frame:
            cheating_frames += 1
            attention_score  = max(0.0, attention_score - ATTENTION_PENALTY)

            # Draw alert banner FIRST
            cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 200), -1)
            short = " | ".join(r[0] for r in reasons)
            cv2.putText(frame, f"ALERT: {short[:80]}", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cheating_pct = (cheating_frames / total_frames * 100) if total_frames else 0

        # ── HUD drawn before saving so screenshot includes it ──
        mouth_act_pct = (len(mouth_activity_log) / (MOUTH_MOVEMENT_WINDOW * 30)) * 100
        draw_hud(frame, blink_count, cheating_count, cheating_pct, attention_score,
                 pitch_v, yaw_v, roll_v, gaze_l_str, gaze_r_str,
                 mouth_gap_v, blink_state_v, extra_body_frames >= EXTRA_BODY_FRAMES,
                 head_yaw_partial_frames, min(mouth_act_pct, 100.0))

        # ── Save screenshot AFTER all drawing — exact match to live view ──
        if cheating_this_frame:
            for reason_text, reason_key in reasons:
                save_screenshot(frame, reason_text, reason_key)

        cv2.imshow('Cheating & Blink Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()

# ─────────────────────────────────────────────────────────────
# SESSION REPORT
# ─────────────────────────────────────────────────────────────
cheating_pct = (cheating_frames / total_frames * 100) if total_frames else 0
print("\n========= SESSION ENDED =========")
print(f"Total Frames          : {total_frames}")
print(f"Total Blinks          : {blink_count}")
print(f"Total Cheating Alerts : {cheating_count}")
print(f"Cheating %            : {cheating_pct:.2f}%")
print(f"Final Attention Score : {attention_score:.1f}%")
print("=================================")

# Save JSON log
with open(log_path, "w") as f:
    json.dump({
        "total_frames"   : total_frames,
        "blinks"         : blink_count,
        "cheating_count" : cheating_count,
        "cheating_pct"   : round(cheating_pct, 2),
        "attention_score": round(attention_score, 1),
        "events"         : event_log,
    }, f, indent=2)
print(f"Session log saved: {log_path}")