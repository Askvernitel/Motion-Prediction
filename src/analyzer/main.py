import threading
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import simpleaudio
import os

# ============ AUDIO / ALARM ============

_alarm_lock = threading.Lock()
_alarm_wave = None
SOUND_PATH = os.path.join(os.getcwd(), "src", "analyzer", "sounds", "sound1.wav")


def playAlarm():
    global _alarm_wave
    if not _alarm_lock.acquire(blocking=False):
        print("Another thread is already playing the sound, skipping.")
        return

    try:
        if _alarm_wave is None:
            _alarm_wave = simpleaudio.WaveObject.from_wave_file(SOUND_PATH)

        print("Playing sound...")
        play_obj = _alarm_wave.play()
        # Block in this background thread so sounds do not overlap
        play_obj.wait_done()
        print("Alarm playback finished.")
    finally:
        _alarm_lock.release()


# ============ LANDMARK CONSTANTS (optional convenience) ============

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

# ============ UPPER BODY ============
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16

# ============ HANDS ============
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22

# ============ TORSO ============
LEFT_HIP = 23
RIGHT_HIP = 24

# ============ LOWER BODY ============
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# ============ FEET ============
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32


class HandTorsoDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose

    def is_hand_near_torso(self, landmarks, threshold=0.05):
        """
        Detects if either hand is near the torso area.

        Args:
            landmarks: pose_landmarks from MediaPipe
            threshold: distance threshold (normalized)

        Returns:
            dict with detection results for left and right hands
        """
        if not landmarks:
            return {"left_hand": False, "right_hand": False}

        lm = landmarks.landmark

        # Hand landmarks
        left_wrist = np.array([
            lm[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
            lm[self.mp_pose.PoseLandmark.LEFT_WRIST].y,
            lm[self.mp_pose.PoseLandmark.LEFT_WRIST].z,
        ])

        right_wrist = np.array([
            lm[self.mp_pose.PoseLandmark.RIGHT_WRIST].x,
            lm[self.mp_pose.PoseLandmark.RIGHT_WRIST].y,
            lm[self.mp_pose.PoseLandmark.RIGHT_WRIST].z,
        ])

        # Torso reference points
        left_shoulder = np.array([
            lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
            lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y,
            lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].z,
        ])

        right_shoulder = np.array([
            lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
            lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
            lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].z,
        ])

        left_hip = np.array([
            lm[self.mp_pose.PoseLandmark.LEFT_HIP].x,
            lm[self.mp_pose.PoseLandmark.LEFT_HIP].y,
            lm[self.mp_pose.PoseLandmark.LEFT_HIP].z,
        ])

        right_hip = np.array([
            lm[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
            lm[self.mp_pose.PoseLandmark.RIGHT_HIP].y,
            lm[self.mp_pose.PoseLandmark.RIGHT_HIP].z,
        ])

        torso_center = (left_shoulder + right_shoulder + left_hip + right_hip) / 4.0

        left_dist = np.linalg.norm(left_wrist[:2] - torso_center[:2])
        right_dist = np.linalg.norm(right_wrist[:2] - torso_center[:2])

        return {
            "left_hand": left_dist < threshold,
            "right_hand": right_dist < threshold,
            "left_distance": float(left_dist),
            "right_distance": float(right_dist),
        }

    def get_torso_region(self, landmarks, margin=0.1):
        """
        Returns the bounding box of the torso region.
        Useful for visualization or more complex checks.
        """
        if not landmarks:
            return None

        lm = landmarks.landmark

        shoulder_left = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        shoulder_right = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        hip_left = lm[self.mp_pose.PoseLandmark.LEFT_HIP]
        hip_right = lm[self.mp_pose.PoseLandmark.RIGHT_HIP]

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
            "max_y": max_y,
        }


class PoseTracker:
    MAX_DIST_KNIFE = 0.1
    DETECT_KNIFE_SEC = 3
    suspiciousCount = 0

    def __init__(self, buffer_size=30, fps=30):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )

        self.NUM_LM = 33
        self.VEC_DIM = 3
        self.FPS = fps
        self.DT = 1.0 / fps
        self.N_FRAMES = buffer_size

        # Buffers
        self.pos_mat = np.zeros((self.N_FRAMES, self.NUM_LM, 3), np.float32)
        self.vel_mat = np.zeros((self.N_FRAMES, self.NUM_LM, self.VEC_DIM), np.float32)
        self.spd_mat = np.zeros((self.N_FRAMES, self.NUM_LM), np.float32)

        self.prev = None
        self.idx = 0
        self.filled = 0

        # Display flags
        self.show_skel = True
        self.show_bbox = True
        self.show_lm = True
        self.show_vec = True

        # Cache
        self.POSE_CONNECTIONS = self.mp_pose.POSE_CONNECTIONS
        self.POSE_LANDMARK = self.mp_pose.PoseLandmark
        self.drawing_utils = mp.solutions.drawing_utils

        # Hand / torso detector reused per frame
        self.hand_detector = HandTorsoDetector()

        # Target poses storage
        self.target_poses = {}
        self.current_target = None

    def get_forward_direction(self):
        """
        Returns the normalized forward-facing direction of the person.
        Computed from torso center -> nose vector.
        """

        positions = self.get_current_positions()
        if positions is None:
            return None

        nose = positions[NOSE]

        left_hip = positions[LEFT_HIP]
        right_hip = positions[RIGHT_HIP]
        torso_center = (left_hip + right_hip) / 2.0

        forward = nose - torso_center  # pointing toward where the person faces
        norm = np.linalg.norm(forward)

        if norm == 0:
            return None

        return forward / norm
    def get_bone_direction(self, id1, id2):
        """
        Returns the normalized direction from id1 -> id2.
        Useful for punch direction (elbow -> wrist).
        """
        pos = self.get_current_positions()
        if pos is None:
            return None

        v = pos[id2] - pos[id1]
        n = np.linalg.norm(v)
        if n == 0:
            return None

        return v / n

    def is_punch_toward_forward(self, wrist_id, elbow_id, angle_threshold_deg=40):
        """
        Check if punch direction matches the person's forward direction.
        """

        forward = self.get_forward_direction()
        punch_dir = self.get_bone_direction(elbow_id, wrist_id)

        if forward is None or punch_dir is None:
            return False

        # Dot product gives cosine of angle
        dot = np.dot(forward, punch_dir)
        dot = np.clip(dot, -1.0, 1.0)

        angle = np.degrees(np.arccos(dot))

        # If angle is small → punch is aimed forward
        return angle < angle_threshold_deg
    def is_punching(self, wrist_id, elbow_id, shoulder_id,
                    min_speed=1.5, min_forward_speed=0.015,
                    min_extension_angle=150):
        """
        Detects a punch by checking:
        1. High wrist speed
        2. Forward movement (decreasing Z)
        3. Almost fully extended arm
        """
        if not self.is_punch_toward_forward(wrist_id, elbow_id):
            return False

        positions = self.get_current_positions()
        velocities = self.get_current_velocities()

        if positions is None or velocities is None:
            return False

        wrist_pos = positions[wrist_id]
        wrist_vel = velocities[wrist_id]

        # 1) Forward motion toward camera (Z decreases)
        forward_speed = -wrist_vel[2]  # negative Z means moving forward
        if forward_speed < min_forward_speed:
            return False

        # 2) Overall speed threshold
        speed = np.linalg.norm(wrist_vel)
        if speed < min_speed:
            return False

        # 3) Arm extension angle at elbow (shoulder-elbow-wrist)
        shoulder = positions[shoulder_id]
        elbow = positions[elbow_id]
        wrist = positions[wrist_id]

        upper_arm = shoulder - elbow
        forearm = wrist - elbow

        n1 = np.linalg.norm(upper_arm)
        n2 = np.linalg.norm(forearm)
        if n1 == 0 or n2 == 0:
            return False

        upper_arm /= n1
        forearm /= n2

        dot = np.dot(upper_arm, forearm)
        dot = np.clip(dot, -1.0, 1.0)
        angle = np.degrees(np.arccos(dot))

        if angle < min_extension_angle:
            return False

        return True
    def get_torso_direction(self):
        """Normalized left→right torso direction, useful for determining shoulder orientation."""
        pos = self.get_current_positions()
        if pos is None:
            return None

        left_shoulder = pos[LEFT_SHOULDER]
        right_shoulder = pos[RIGHT_SHOULDER]

        v = right_shoulder - left_shoulder
        n = np.linalg.norm(v)
        if n == 0:
            return None

        return v / n

    def get_head_direction(self):
        """Normalized head facing direction using eyes + nose."""
        pos = self.get_current_positions()
        if pos is None:
            return None

        left_eye = pos[LEFT_EYE]
        right_eye = pos[RIGHT_EYE]
        eye_center = (left_eye + right_eye) / 2.0
        nose = pos[NOSE]

        v = nose - eye_center
        n = np.linalg.norm(v)
        if n == 0:
            return None

        return v / n
    def is_checking_over_shoulder(self, angle_threshold_deg=40):
        """
        Returns:
        'left'  → looking over left shoulder
        'right' → looking over right shoulder
        None    → not checking
        """

        head_dir = self.get_head_direction()
        torso_dir = self.get_torso_direction()

        if head_dir is None or torso_dir is None:
            return None

        # angle between head facing direction and torso left→right axis
        dot = np.dot(head_dir[:2], torso_dir[:2])  # ignore z for cleaner yaw
        dot = np.clip(dot, -1.0, 1.0)

        angle = np.degrees(np.arccos(dot))

        if angle < angle_threshold_deg:
            return None

        # determine direction: left or right
        cross = head_dir[0] * torso_dir[1] - head_dir[1] * torso_dir[0]

        if cross > 0:
            return "left"
        else:
            return "right"
    # ============ KNIFE / SUSPICIOUS DISTANCE ============

    def has_knife(self, pos1, pos2):
        if pos1 is None or pos2 is None:
            return False
        return self.distance_numpy(pos1, pos2)

    def distance_numpy(self, pos1, pos2):
        pos1 = np.array(pos1[:2])
        pos2 = np.array(pos2[:2])
        dist = np.linalg.norm(pos1 - pos2)

        if dist < self.MAX_DIST_KNIFE:
            self.suspiciousCount += 1
        else:
            self.suspiciousCount = 0

        elapsed = self.suspiciousCount / self.FPS
        print("SUS:", elapsed)

        return elapsed >= self.DETECT_KNIFE_SEC

    # ============ HISTORY ACCESSORS ============

    def get_current_positions(self):
        if self.filled == 0:
            return None
        pid = (self.idx - 1) % self.N_FRAMES
        return self.pos_mat[pid].copy()

    def get_current_velocities(self):
        if self.filled == 0:
            return None
        pid = (self.idx - 1) % self.N_FRAMES
        return self.vel_mat[pid].copy()

    def get_current_speeds(self):
        if self.filled == 0:
            return None
        pid = (self.idx - 1) % self.N_FRAMES
        return self.spd_mat[pid].copy()

    def get_position_history(self, n_frames=None):
        if self.filled == 0:
            return None
        n = n_frames if n_frames else self.filled
        n = min(n, self.filled)

        indices = [(self.idx - i - 1) % self.N_FRAMES for i in range(n)]
        return self.pos_mat[indices].copy()

    def get_velocity_history(self, n_frames=None):
        if self.filled == 0:
            return None
        n = n_frames if n_frames else self.filled
        n = min(n, self.filled)

        indices = [(self.idx - i - 1) % self.N_FRAMES for i in range(n)]
        return self.vel_mat[indices].copy()

    # ============ PER-LANDMARK HELPERS ============

    def get_landmark_position(self, landmark_id):
        pos = self.get_current_positions()
        if pos is None:
            return None
        return pos[landmark_id]

    def get_landmark_velocity(self, landmark_id):
        vel = self.get_current_velocities()
        if vel is None:
            return None
        return vel[landmark_id]

    def get_average_speed(self):
        spd = self.get_current_speeds()
        if spd is None:
            return 0.0
        return float(np.mean(spd))

    def get_max_speed_landmark(self):
        spd = self.get_current_speeds()
        if spd is None:
            return None
        return int(np.argmax(spd))

    # ============ POSE SCORING SYSTEM ============

    def save_target_pose(self, pose_name):
        pos = self.get_current_positions()
        if pos is not None:
            self.target_poses[pose_name] = pos.copy()
            return True
        return False

    def set_target_pose(self, pose_name):
        if pose_name in self.target_poses:
            self.current_target = pose_name
            return True
        return False

    def calculate_pose_distance(self, target_pose):
        current = self.get_current_positions()
        if current is None or target_pose is None:
            return None

        distances = np.linalg.norm(current - target_pose, axis=1)
        return distances

    def calculate_stability_score(self, n_frames=10):
        vel_hist = self.get_velocity_history(n_frames)
        if vel_hist is None or len(vel_hist) < 2:
            return 0.0

        speeds = np.linalg.norm(vel_hist, axis=2)
        avg_speed = np.mean(speeds)

        stability = max(0, 100 - (avg_speed * 20))
        return float(min(100, stability))

    def calculate_pose_score(
        self,
        pose_name=None,
        position_weight=0.6,
        stability_weight=0.4,
        n_stability_frames=15,
    ):
        target_name = pose_name or self.current_target

        if target_name not in self.target_poses:
            return None

        target = self.target_poses[target_name]
        distances = self.calculate_pose_distance(target)

        if distances is None:
            return None

        avg_distance = np.mean(distances)
        position_score = max(0, 100 - (avg_distance * 1000))
        position_score = float(min(100, position_score))

        stability_score = self.calculate_stability_score(n_stability_frames)

        total_score = (
            position_score * position_weight + stability_score * stability_weight
        )

        return {
            "total_score": total_score,
            "position_score": position_score,
            "stability_score": stability_score,
            "avg_distance": float(avg_distance),
            "distances": distances,
        }

    def is_pose_locked(
        self,
        pose_name=None,
        position_threshold=90,
        stability_threshold=85,
        n_frames=10,
    ):
        scores = self.calculate_pose_score(
            pose_name=pose_name, n_stability_frames=n_frames
        )

        if scores is None:
            return False

        return (
            scores["position_score"] >= position_threshold
            and scores["stability_score"] >= stability_threshold
        )

    def calculate_distance_3d(self, landmark_id_1, landmark_id_2):
        positions = self.get_current_positions()
        if positions is None:
            return None

        p1 = positions[landmark_id_1]
        p2 = positions[landmark_id_2]

        diff = p1 - p2
        dist = np.linalg.norm(diff)
        return float(dist)
    def calculate_distance_2d(self, landmark_id_1, landmark_id_2):
        positions = self.get_current_positions()
        if positions is None:
            return None

        point1 = positions[landmark_id_1]
        point2 = positions[landmark_id_2]

        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]

        distance = (dx ** 2 + dy ** 2) ** 0.5
        print(distance)
        return float(distance)

    def get_worst_landmarks(self, pose_name=None, n=5):
        target_name = pose_name or self.current_target

        if target_name not in self.target_poses:
            return None

        target = self.target_poses[target_name]
        distances = self.calculate_pose_distance(target)

        if distances is None:
            return None

        worst_indices = np.argsort(distances)[-n:][::-1]

        return [(int(idx), float(distances[idx])) for idx in worst_indices]

    # ============ MAIN FRAME PROCESSING ============

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        red_spec = self.drawing_utils.DrawingSpec(
            color=(0, 0, 255), thickness=2, circle_radius=2
        )

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            detection = self.hand_detector.is_hand_near_torso(
                res.pose_landmarks, threshold=0.15
            )

            arr = np.fromiter(
                (v for p in lm for v in (p.x, p.y, p.z)),
                dtype=np.float32,
            ).reshape(self.NUM_LM, 3)

            self.pos_mat[self.idx] = arr

            if self.prev is not None:
                dv = (arr - self.prev) * (1.0 / self.DT)
                self.vel_mat[self.idx] = dv
                self.spd_mat[self.idx] = np.linalg.norm(dv, axis=1)
            self.prev = arr

            self.idx = (self.idx + 1) % self.N_FRAMES
            self.filled = min(self.filled + 1, self.N_FRAMES)

            xs = arr[:, 0] * w
            ys = arr[:, 1] * h

            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())

            if self.show_bbox:
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

            if self.show_skel:
                self.drawing_utils.draw_landmarks(
                    frame, res.pose_landmarks, self.POSE_CONNECTIONS
                )

            pts = None
            if self.show_lm:
                pts = np.column_stack((xs.astype(int), ys.astype(int)))
                for p in pts:
                    cv2.circle(frame, tuple(p), 3, (255, 0, 0), -1)
            else:
                pts = np.column_stack((xs.astype(int), ys.astype(int)))

            if detection["left_hand"] or detection["right_hand"]:
                threading.Thread(target=playAlarm, daemon=True).start()

            # Example knife check (still commented out by default)
            """
            if self.has_knife(self.get_landmark_position(LEFT_HIP),
                              self.get_landmark_position(LEFT_INDEX)):
                self.drawing_utils.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    self.POSE_CONNECTIONS,
                    landmark_drawing_spec=red_spec,
                    connection_drawing_spec=red_spec
                )
            """

            if self.show_vec and self.prev is not None:
                pid = (self.idx - 1) % self.N_FRAMES
                dv = self.vel_mat[pid]
                vx = (dv[:, 0] * 500).astype(int)
                vy = (dv[:, 1] * 500).astype(int)

                for k, (px, py) in enumerate(pts):
                    if abs(vx[k]) > 2 or abs(vy[k]) > 2:
                        end = (px + vx[k], py + vy[k])
                        cv2.arrowedLine(frame, (px, py), end, (0, 255, 0), 2)

        return frame

    def release(self):
        self.pose.close()


def run():
    tracker = PoseTracker(buffer_size=30, fps=30)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mode = "scoring"

    print("=== POSE SCORING SYSTEM ===")
    print("Commands:")
    print("  's' - Save current pose as target")
    print("  't' - Switch to scoring mode")
    print("  'f' - Switch to freeform mode")
    print("  'q' - Quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        frame = tracker.process_frame(frame)

        if mode == "scoring" and tracker.current_target:
            scores = tracker.calculate_pose_score()

            if scores:
                y_offset = 30
                cv2.putText(
                    frame,
                    f"Mode: SCORING ({tracker.current_target})",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                y_offset += 35
                cv2.putText(
                    frame,
                    f"Total Score: {scores['total_score']:.1f}/100",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                y_offset += 30
                cv2.putText(
                    frame,
                    f"Position: {scores['position_score']:.1f}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 200, 100),
                    2,
                )

                y_offset += 30
                cv2.putText(
                    frame,
                    f"Stability: {scores['stability_score']:.1f}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (100, 200, 255),
                    2,
                )

                is_locked = tracker.is_pose_locked()
                if is_locked:
                    y_offset += 35
                    cv2.putText(
                        frame,
                        "POSE LOCKED!",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        3,
                    )

                worst = tracker.get_worst_landmarks(n=3)
                if worst and scores["position_score"] < 90:
                    y_offset += 35
                    cv2.putText(
                        frame,
                        "Fix landmarks:",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 100, 255),
                        1,
                    )
                    for idx, dist in worst:
                        y_offset += 20
                        cv2.putText(
                            frame,
                            f"  #{idx}: {dist:.3f}",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 150, 255),
                            1,
                        )
        else:
            cv2.putText(
                frame,
                "Mode: FREEFORM (Press 's' to save pose)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )

            avg_spd = tracker.get_average_speed()
            cv2.putText(
                frame,
                f"Avg Speed: {avg_spd:.2f}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Pose Scoring System", frame)
        k = cv2.waitKey(1) & 0xFF

        if k == ord("q"):
            break
        elif k == ord("s"):
            if tracker.save_target_pose("target_pose"):
                tracker.set_target_pose("target_pose")
                print("✓ Target pose saved!")
                mode = "scoring"
        elif k == ord("t"):
            mode = "scoring"
            print("Switched to scoring mode")
        elif k == ord("f"):
            mode = "freeform"
            print("Switched to freeform mode")

    cap.release()
    cv2.destroyAllWindows()
    tracker.release()


if __name__ == "__main__":
    run()

