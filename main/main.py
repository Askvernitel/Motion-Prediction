import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

NUM_LM = 33
VEC_DIM = 3
FPS = 30
DT = 1.0 / FPS

N_FRAMES = 30

pos_mat = np.zeros((N_FRAMES, NUM_LM, 3), np.float32)
vel_mat = np.zeros((N_FRAMES, NUM_LM, VEC_DIM), np.float32)
spd_mat = np.zeros((N_FRAMES, NUM_LM), np.float32)

prev = None
idx = 0
filled = 0

show_skel = True
show_bbox = True
show_lm = True
show_vec = True

# Pre-reference to avoid attribute lookups in hot loop
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS
POSE_LANDMARK = mp_pose.PoseLandmark

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark

        # Convert directly into NumPy in one pass (no Python loop per coordinate)
        arr = np.fromiter((v for p in lm for v in (p.x, p.y, p.z)),
                          dtype=np.float32).reshape(NUM_LM, 3)

        pos_mat[idx] = arr

        if prev is not None:
            dv = (arr - prev) * (1.0 / DT)
            vel_mat[idx] = dv
            spd_mat[idx] = np.linalg.norm(dv, axis=1)
        prev = arr

        idx = (idx + 1) % N_FRAMES
        filled = min(filled + 1, N_FRAMES)

        xs = arr[:, 0] * w
        ys = arr[:, 1] * h

        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

        if show_bbox:
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0,255,0), 2)

        if show_skel:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, res.pose_landmarks, POSE_CONNECTIONS)

        if show_lm:
            pts = np.column_stack((xs.astype(int), ys.astype(int)))
            for p in pts:
                cv2.circle(frame, tuple(p), 3, (255,0,0), -1)

        if show_vec and prev is not None:
            pid = (idx - 1) % N_FRAMES
            dv = vel_mat[pid]
            vx = (dv[:,0] * 500).astype(int)
            vy = (dv[:,1] * 500).astype(int)

            for k, (px, py) in enumerate(pts):
                if abs(vx[k]) > 2 or abs(vy[k]) > 2:
                    end = (px + vx[k], py + vy[k])
                    cv2.arrowedLine(frame, (px, py), end, (0,255,0), 2)

    cv2.imshow("pose", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
