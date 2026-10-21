"""
Camera-based motion analysis for squat / jump / running.

Pipeline:
  1. OpenCV reads the video frame by frame.
  2. MediaPipe Pose detects 33 body landmarks per frame.
  3. Joint angles / vertical displacement are computed from those landmarks.
  4. Movement-specific metrics are extracted (squat depth, jump height, stride rate).
  5. An annotated video + a CSV of per-frame metrics + a plot are saved.

Usage:
  python motion_analysis.py --video squat.mp4 --mode squat
  python motion_analysis.py --video jump.mp4  --mode jump  --ref_height_cm 175
  python motion_analysis.py --video run.mp4   --mode run
  python motion_analysis.py --video 0         --mode squat   # webcam
"""

import argparse
import csv
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


def angle(a, b, c):
    """Angle at vertex b (in degrees), given 3 (x,y) points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos_ang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))


def get_xy(landmarks, idx, w, h):
    lm = landmarks[idx]
    return [lm.x * w, lm.y * h]


def smooth(series, window=5):
    if len(series) < window:
        return series
    kernel = np.ones(window) / window
    return list(np.convolve(series, kernel, mode="same"))


def analyze(video_path, mode, ref_height_cm, out_video, out_csv, out_plot):
    cap = cv2.VideoCapture(int(video_path) if video_path.isdigit() else video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    L = mp_pose.PoseLandmark
    rows = []  # per-frame metrics
    hip_y_series, knee_angle_series, ankle_y_series = [], [], []
    px_to_cm = None  # calibration factor, set from reference height if given

    with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5,
                       min_tracking_confidence=0.5) as pose:
        frame_idx = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                lms = result.pose_landmarks.landmark
                mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                hip = get_xy(lms, L.LEFT_HIP, w, h)
                knee = get_xy(lms, L.LEFT_KNEE, w, h)
                ankle = get_xy(lms, L.LEFT_ANKLE, w, h)
                shoulder = get_xy(lms, L.LEFT_SHOULDER, w, h)
                r_ankle = get_xy(lms, L.RIGHT_ANKLE, w, h)
                r_hip = get_xy(lms, L.RIGHT_HIP, w, h)

                knee_angle = angle(hip, knee, ankle)
                hip_angle = angle(shoulder, hip, knee)
                hip_mid_y = (hip[1] + r_hip[1]) / 2

                # one-time calibration: pixel height of person -> cm
                if px_to_cm is None and ref_height_cm:
                    top = get_xy(lms, L.NOSE, w, h)
                    bottom_y = max(ankle[1], r_ankle[1])
                    px_height = bottom_y - top[1]
                    if px_height > 0:
                        px_to_cm = ref_height_cm / px_height

                hip_y_series.append(hip_mid_y)
                knee_angle_series.append(knee_angle)
                ankle_y_series.append((ankle[1] + r_ankle[1]) / 2)

                rows.append({
                    "frame": frame_idx,
                    "time_s": round(frame_idx / fps, 3),
                    "knee_angle_deg": round(knee_angle, 1),
                    "hip_angle_deg": round(hip_angle, 1),
                    "hip_y_px": round(hip_mid_y, 1),
                    "ankle_y_px": round((ankle[1] + r_ankle[1]) / 2, 1),
                })

                cv2.putText(frame, f"Knee: {knee_angle:.0f} deg", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Hip: {hip_angle:.0f} deg", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            writer.write(frame)
            frame_idx += 1

    cap.release()
    writer.release()

    # write CSV
    with open(out_csv, "w", newline="") as f:
        if rows:
            w_ = csv.DictWriter(f, fieldnames=rows[0].keys())
            w_.writeheader()
            w_.writerows(rows)

    metrics = {}
    knee_s = smooth(knee_angle_series)
    hip_s = smooth(hip_y_series)
    ankle_s = smooth(ankle_y_series)

    if mode == "squat":
        metrics["min_knee_angle_deg"] = round(min(knee_s), 1)
        metrics["squat_depth_note"] = (
            "deep (<90deg)" if min(knee_s) < 90 else
            "parallel-ish (90-110deg)" if min(knee_s) < 110 else
            "shallow (>110deg)"
        )

    elif mode == "jump":
        baseline = np.median(hip_s[:int(fps)])  # standing baseline, first ~1s
        min_hip_y = min(hip_s)  # smallest y = highest point (image coords: y grows downward)
        rise_px = baseline - min_hip_y
        metrics["jump_rise_px"] = round(rise_px, 1)
        if px_to_cm:
            metrics["jump_height_cm"] = round(rise_px * px_to_cm, 1)
        else:
            metrics["note"] = "pass --ref_height_cm for real-world jump height"

    elif mode == "run":
        # find local minima in ankle_y (foot near ground = lowest point = max y actually,
        # since y grows downward; a foot strike is a local MAX in ankle_y)
        strikes = []
        for i in range(2, len(ankle_s) - 2):
            if ankle_s[i] > ankle_s[i-1] and ankle_s[i] > ankle_s[i+1] and ankle_s[i] > np.mean(ankle_s):
                strikes.append(i)
        # de-duplicate strikes that are too close together (<0.2s apart)
        clean = []
        for s in strikes:
            if not clean or (s - clean[-1]) / fps > 0.2:
                clean.append(s)
        intervals = np.diff(clean) / fps
        metrics["detected_foot_strikes"] = len(clean)
        if len(intervals) > 0:
            metrics["avg_stride_time_s"] = round(float(np.mean(intervals)), 3)
            metrics["cadence_strides_per_min"] = round(60 / np.mean(intervals), 1)

    # plot
    plt.figure(figsize=(9, 4))
    plt.plot(knee_s, label="Knee angle (deg)")
    plt.plot(np.array(hip_s) / max(hip_s) * max(knee_s), label="Hip Y (scaled)", alpha=0.6)
    plt.xlabel("Frame")
    plt.legend()
    plt.title(f"Motion analysis — {mode}")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=120)

    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="path to video file, or '0' for webcam")
    ap.add_argument("--mode", choices=["squat", "jump", "run"], required=True)
    ap.add_argument("--ref_height_cm", type=float, default=None,
                     help="person's real height in cm, for pixel->cm calibration")
    ap.add_argument("--out_video", default="annotated.mp4")
    ap.add_argument("--out_csv", default="metrics.csv")
    ap.add_argument("--out_plot", default="plot.png")
    args = ap.parse_args()

    results = analyze(args.video, args.mode, args.ref_height_cm,
                       args.out_video, args.out_csv, args.out_plot)
    print("\n--- Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")
    print(f"\nSaved: {args.out_video}, {args.out_csv}, {args.out_plot}")