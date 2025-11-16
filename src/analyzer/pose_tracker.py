import threading
import simpleaudio
import numpy as np
import mediapipe as mp
import os

# ============================================================
# CONSTANTS FOR LANDMARK IDs
# ============================================================

NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32


# ============================================================
# AUDIO ALARM (simpleaudio)
# ============================================================

_alarm_lock = threading.Lock()
_alarm_wave = None
SOUND_PATH = os.path.join(os.getcwd(), "src", "analyzer", "sounds", "sound1.wav")


def playAlarm():
    global _alarm_wave

    if not _alarm_lock.acquire(blocking=False):
        return

    try:
        if _alarm_wave is None:
            _alarm_wave = simpleaudio.WaveObject.from_wave_file(SOUND_PATH)

        obj = _alarm_wave.play()
        obj.wait_done()

    finally:
        _alarm_lock.release()


# ============================================================
# HAND–TORSO DETECTOR
# ============================================================
class HandTorsoDetector:

    def __init__(self):
        self.mp_pose = mp.solutions.pose

    def is_hand_near_torso(self, landmarks, threshold=0.05):
        """
        Detects if either hand is near the torso area.

        Args:
            landmarks: pose_landmarks from MediaPipe
            threshold: distance threshold (normalized, default 0.15)

        Returns:
            dict with detection results for left and right hands
        """
        if not landmarks:
            return {"left_hand": False, "right_hand": False}

        # Get key landmark positions
        lm = landmarks.landmark

        # Hand landmarks
        left_wrist = np.array([lm[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
                               lm[self.mp_pose.PoseLandmark.LEFT_WRIST].y,
                               lm[self.mp_pose.PoseLandmark.LEFT_WRIST].z])

        right_wrist = np.array([lm[self.mp_pose.PoseLandmark.RIGHT_WRIST].x,
                                lm[self.mp_pose.PoseLandmark.RIGHT_WRIST].y,
                                lm[self.mp_pose.PoseLandmark.RIGHT_WRIST].z])

        # Torso reference points (shoulders and hips for torso bounding)
        left_shoulder = np.array([lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                  lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                                  lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].z])

        right_shoulder = np.array([lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                   lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                                   lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].z])

        left_hip = np.array([lm[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                             lm[self.mp_pose.PoseLandmark.LEFT_HIP].y,
                             lm[self.mp_pose.PoseLandmark.LEFT_HIP].z])

        right_hip = np.array([lm[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
                              lm[self.mp_pose.PoseLandmark.RIGHT_HIP].y,
                              lm[self.mp_pose.PoseLandmark.RIGHT_HIP].z])

        # Define torso center and bounds
        torso_center = (left_shoulder + right_shoulder + left_hip + right_hip) / 4

        # Calculate distances (using 2D for faster computation)
        left_dist = np.linalg.norm(left_wrist[:2] - torso_center[:2])
        right_dist = np.linalg.norm(right_wrist[:2] - torso_center[:2])

        return {
            "left_hand": left_dist < threshold,
            "right_hand": right_dist < threshold,
            "left_distance": float(left_dist),
            "right_distance": float(right_dist)
        }

    def get_torso_region(self, landmarks, margin=0.05):
        """
        Returns the bounding box of the torso region.
        Useful for visualization or more complex checks.
        """
        if not landmarks:
            return None

        lm = landmarks.landmark

        # Get torso corners
        shoulder_left = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        shoulder_right = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        hip_left = lm[self.mp_pose.PoseLandmark.LEFT_HIP]
        hip_right = lm[self.mp_pose.PoseLandmark.RIGHT_HIP]

        # Calculate bounding box with margin
        x_coords = [shoulder_left.x, shoulder_right.x, hip_left.x, hip_right.x]
        y_coords = [shoulder_left.y, shoulder_right.y, hip_left.y, hip_right.y]

        min_x = min(x_coords) - margin
        max_x = max(x_coords) + margin
        min_y = min(y_coords) - margin
        max_y = max(y_coords) + margin

        return {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y
        }


# ============================================================
# POSE TRACKER CLASS
# ============================================================

class PoseTracker:
    def __init__(self, buffer_size=30, fps=30):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )

        self.NUM_LM = 33
        self.FPS = fps
        self.DT = 1.0 / fps
        self.N_FRAMES = buffer_size

        self.pos_mat = np.zeros((self.N_FRAMES, self.NUM_LM, 3), np.float32)
        self.vel_mat = np.zeros((self.N_FRAMES, self.NUM_LM, 3), np.float32)
        self.spd_mat = np.zeros((self.N_FRAMES, self.NUM_LM), np.float32)

        self.prev = None
        self.idx = 0
        self.filled = 0

        self.hand_detector = HandTorsoDetector()

    # ============================================================
    # LANDMARK ACCESS
    # ============================================================

    def get_current_positions(self):
        if self.filled == 0:
            return None
        return self.pos_mat[(self.idx - 1) % self.N_FRAMES].copy()

    def get_current_velocities(self):
        if self.filled == 0:
            return None
        return self.vel_mat[(self.idx - 1) % self.N_FRAMES].copy()

    def get_landmark_position(self, lid):
        pos = self.get_current_positions()
        if pos is None:
            return None
        return pos[lid]

    def get_landmark_velocity(self, lid):
        vel = self.get_current_velocities()
        if vel is None:
            return None
        return vel[lid]

    # ============================================================
    # FRAME INGEST (NO DISPLAY)
    # ============================================================

    def process_frame_no_draw(self, frame):
        """Update pose buffers from a BGR frame. Returns pose_landmarks."""
        rgb = frame[:, :, ::-1]  # BGR → RGB
        res = self.pose.process(rgb)

        if not res.pose_landmarks:
            return None

        lm = res.pose_landmarks.landmark
        arr = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        self.pos_mat[self.idx] = arr

        if self.prev is not None:
            dv = (arr - self.prev) * (1.0 / self.DT)
            self.vel_mat[self.idx] = dv
            self.spd_mat[self.idx] = np.linalg.norm(dv, axis=1)

        self.prev = arr
        self.idx = (self.idx + 1) % self.N_FRAMES
        self.filled = min(self.filled + 1, self.N_FRAMES)

        return res.pose_landmarks

    # ============================================================
    # DISTANCE / ANGLES
    # ============================================================

    def calculate_distance_2d(self, id1, id2):
        pos = self.get_current_positions()
        if pos is None:
            return None
        p1, p2 = pos[id1], pos[id2]
        return float(np.linalg.norm(p1[:2] - p2[:2]))

    def calculate_distance_3d(self, id1, id2):
        pos = self.get_current_positions()
        if pos is None:
            return None
        return float(np.linalg.norm(pos[id1] - pos[id2]))

    def calculate_bone_angle(self, id1, id2, id3, id4):
        """
        Angle between bones id1->id2 and id3->id4 in 3D.
        """
        pos = self.get_current_positions()
        if pos is None:
            return None

        b1 = pos[id2] - pos[id1]
        b2 = pos[id4] - pos[id3]

        n1 = np.linalg.norm(b1)
        n2 = np.linalg.norm(b2)
        if n1 == 0 or n2 == 0:
            return None

        b1 /= n1
        b2 /= n2

        dot = np.clip(np.dot(b1, b2), -1.0, 1.0)
        return float(np.degrees(np.arccos(dot)))

    # ============================================================
    # ORIENTATION HELPERS
    # ============================================================

    def get_head_yaw(self):
        """
        2D head yaw direction: nose relative to eye center.
        """
        pos = self.get_current_positions()
        if pos is None:
            return None

        left_eye = pos[LEFT_EYE][:2]
        right_eye = pos[RIGHT_EYE][:2]
        eye_center = (left_eye + right_eye) / 2.0

        nose = pos[NOSE][:2]
        yaw = nose - eye_center

        n = np.linalg.norm(yaw)
        return yaw / n if n > 0 else None

    def get_forward_direction(self):
        """
        3D forward direction from torso center to nose.
        Used for punch direction.
        """
        pos = self.get_current_positions()
        if pos is None:
            return None

        nose = pos[NOSE]
        torso = (pos[LEFT_HIP] + pos[RIGHT_HIP]) / 2.0
        v = nose - torso
        n = np.linalg.norm(v)
        return v / n if n > 0 else None

    def get_torso_side(self):
        """
        2D left→right torso axis (shoulder line).
        """
        pos = self.get_current_positions()
        if pos is None:
            return None

        left_sh = pos[LEFT_SHOULDER][:2]
        right_sh = pos[RIGHT_SHOULDER][:2]

        v = right_sh - left_sh
        n = np.linalg.norm(v)
        return v / n if n > 0 else None

    # ============================================================
    # CHECKING OVER SHOULDER
    # ============================================================

    def is_checking_over_shoulder(self,
                                min_angle_deg=40,
                                min_offset_norm=0.08):
        """
        Detect looking over shoulder only when BOTH are true:
        - head yaw angle exceeds min_angle_deg
        - nose offset from shoulder center exceeds min_offset_norm
        """

        pos = self.get_current_positions()
        if pos is None:
            return None

        # ------------------------------
        # 1. Compute yaw angle
        # ------------------------------
        left_eye = pos[LEFT_EYE][:2]
        right_eye = pos[RIGHT_EYE][:2]
        eye_center = (left_eye + right_eye) / 2.0
        nose = pos[NOSE][:2]

        yaw_vec = nose - eye_center
        yaw_norm = np.linalg.norm(yaw_vec)
        if yaw_norm == 0:
            return None

        yaw_vec /= yaw_norm

        # head left-right axis (eye to eye)
        head_lr = right_eye - left_eye
        lr_norm = np.linalg.norm(head_lr)
        if lr_norm == 0:
            return None

        head_lr /= lr_norm  # normalized left→right axis

        # yaw angle relative to straight-ahead
        dot = np.clip(np.dot(yaw_vec, head_lr), -1.0, 1.0)
        yaw_angle = abs(np.degrees(np.arccos(dot)))

        if yaw_angle < min_angle_deg:
            return None  # head not rotated enough

        # ------------------------------
        # 2. Nose lateral offset relative to shoulders
        # ------------------------------
        left_sh = pos[LEFT_SHOULDER][:2]
        right_sh = pos[RIGHT_SHOULDER][:2]

        shoulder_mid = (left_sh + right_sh) / 2
        shoulder_span = np.linalg.norm(right_sh - left_sh)
        if shoulder_span == 0:
            return None

        offset = (nose - shoulder_mid)[0]            # horizontal offset
        offset_norm = offset / shoulder_span         # normalize by torso width

        if offset_norm > min_offset_norm:
            return "right"
        if offset_norm < -min_offset_norm:
            return "left"

        return None
    # ============================================================
    # PUNCHING DETECTION
    # ============================================================

    def is_punching(self, wrist, elbow, shoulder,
                    min_speed=1.5, min_forward_speed=0.015,
                    min_extension_angle=150):
        pos = self.get_current_positions()
        vel = self.get_current_velocities()

        if pos is None or vel is None:
            return False

        wrist_vel = vel[wrist]

        # Forward motion toward camera: -Z
        forward_speed = -wrist_vel[2]
        if forward_speed < min_forward_speed:
            return False

        # Speed threshold
        speed = np.linalg.norm(wrist_vel)
        if speed < min_speed:
            return False

        # Arm extension angle at elbow
        pS, pE, pW = pos[shoulder], pos[elbow], pos[wrist]
        upper = pS - pE
        fore = pW - pE

        nu = np.linalg.norm(upper)
        nf = np.linalg.norm(fore)
        if nu == 0 or nf == 0:
            return False

        upper /= nu
        fore /= nf

        dot = np.clip(np.dot(upper, fore), -1.0, 1.0)
        angle = np.degrees(np.arccos(dot))

        if angle < min_extension_angle:
            return False

        return True

    def is_punch_toward_forward(self, wrist, elbow, angle_threshold_deg=40):
        forward = self.get_forward_direction()
        pos = self.get_current_positions()
        if forward is None or pos is None:
            return False

        pE, pW = pos[elbow], pos[wrist]
        punch = pW - pE
        n = np.linalg.norm(punch)
        if n == 0:
            return False

        punch /= n
        dot = np.clip(np.dot(forward, punch), -1.0, 1.0)
        angle = np.degrees(np.arccos(dot))

        return angle < angle_threshold_deg

    # ============================================================
    # CLEANUP
    # ============================================================

    def release(self):
        self.pose.close()
