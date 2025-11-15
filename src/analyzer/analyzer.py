import threading

import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import cv2
import mediapipe as mp
import numpy as np
from playsound import playsound
import errors.error_handler as err
import os
# Hand is near torso - do something

# ============ FACE ============

lock = threading.Lock()
def playAlarm():
    if lock.acquire(blocking=False):
        try:
            print("Playing sound...")
            playsound(os.getcwd() + "/src" + "/analyzer" + "/sounds" + "/sound1.mp3")
        finally:
            lock.release()
    else:
        print("Another thread is already playing the sound, skipping.")
    print("Starting asynchronous task...")
    print("Asynchronous task finished.")
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


class PoseTracker:
    MAX_DIST_KNIFE = 0.1
    DETECT_KNIFE_SEC = 3
    suspiciousCount = 0
    def has_knife(self, pos1, pos2):

        return self.distance_numpy(pos1,pos2)
    # from datetime import datetime
    def distance_numpy(self,pos1, pos2):
        pos1 = np.array(pos1[0:2])
        pos2 = np.array(pos2[0:2])
        dist = np.linalg.norm(pos1 - pos2)
        print("SUS", self.suspiciousCount//self.FPS)
        if((self.suspiciousCount // self.FPS) >= self.DETECT_KNIFE_SEC):
            return True
        if (dist < self.MAX_DIST_KNIFE):
            self.suspiciousCount += 1
        return False

    def __init__(self, buffer_size=30, fps=30, error_counter:err.ErrorCounter=None):
        self.mp_pose = mp.solutions.pose
        self.error_counter = error_counter
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
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

        # Target poses storage
        self.target_poses = {}
        self.current_target = None

    def get_current_positions(self):
        """Get current frame positions (Nx3: x, y, z)"""
        if self.filled == 0:
            return None
        pid = (self.idx - 1) % self.N_FRAMES
        return self.pos_mat[pid].copy()

    def get_current_velocities(self):
        """Get current frame velocities (Nx3: vx, vy, vz)"""
        if self.filled == 0:
            return None
        pid = (self.idx - 1) % self.N_FRAMES
        return self.vel_mat[pid].copy()

    def get_current_speeds(self):
        """Get current frame speeds (N,)"""
        if self.filled == 0:
            return None
        pid = (self.idx - 1) % self.N_FRAMES
        return self.spd_mat[pid].copy()

    def is_position_locked(self, threshold=0.02, frame_window=5):
        """
        Determine if the person's position is locked (stationary)

        Args:
            threshold: Maximum average movement allowed to consider locked (0.02 = 2% of frame)
            frame_window: Number of recent frames to analyze

        Returns:
            bool: True if position is locked, False otherwise
        """
        if not self.get_position_history(20) or len(self.get_position_history(20)) < 2:
            return False
        print(self.get_position_history(20))
        # Get recent frames to analyze
        start_idx = max(0, len(self.get_position_history(20)) - frame_window)
        recent_frames = self.get_position_history(20)[start_idx:]

        # Filter out frames without landmarks
        valid_frames = [f for f in recent_frames if f['landmarks'] is not None]

        if len(valid_frames) < 2:
            return False

        # Calculate movement between consecutive frames
        total_movement = 0
        movement_count = 0

        for i in range(1, len(valid_frames)):
            prev_landmarks = valid_frames[i - 1]['landmarks']
            curr_landmarks = valid_frames[i]['landmarks']

            # Calculate average movement across all landmarks
            frame_movement = 0
            for prev_point, curr_point in zip(prev_landmarks, curr_landmarks):
                dx = curr_point[0] - prev_point[0]
                dy = curr_point[1] - prev_point[1]
                distance = (dx ** 2 + dy ** 2) ** 0.5
                frame_movement += distance

            # Average movement per landmark
            avg_movement = frame_movement / len(prev_landmarks)
            total_movement += avg_movement
            movement_count += 1

        # Calculate overall average movement
        overall_avg_movement = total_movement / movement_count

        # Position is locked if average movement is below threshold
        return overall_avg_movement < threshold
    def get_position_history(self, n_frames=None):
        """Get last n frames of positions (n x N x 3)"""
        if self.filled == 0:
            return None
        n = n_frames if n_frames else self.filled
        n = min(n, self.filled)

        indices = [(self.idx - i - 1) % self.N_FRAMES for i in range(n)]
        return self.pos_mat[indices].copy()

    def get_velocity_history(self, n_frames=None):
        """Get last n frames of velocities (n x N x 3)"""
        if self.filled == 0:
            return None
        n = n_frames if n_frames else self.filled
        n = min(n, self.filled)

        indices = [(self.idx - i - 1) % self.N_FRAMES for i in range(n)]
        return self.vel_mat[indices].copy()

    def get_landmark_position(self, landmark_id):
        """Get current position of specific landmark (3,)"""
        pos = self.get_current_positions()
        if pos is None:
            return None
        return pos[landmark_id]

    def get_landmark_velocity(self, landmark_id):
        """Get current velocity of specific landmark (3,)"""
        vel = self.get_current_velocities()
        if vel is None:
            return None
        return vel[landmark_id]

    def get_average_speed(self):
        """Get average speed across all landmarks"""
        spd = self.get_current_speeds()
        if spd is None:
            return 0.0
        return float(np.mean(spd))

    def get_max_speed_landmark(self):
        """Get landmark ID with maximum speed"""
        spd = self.get_current_speeds()
        if spd is None:
            return None
        return int(np.argmax(spd))

    # ============ POSE SCORING SYSTEM ============

    def save_target_pose(self, pose_name):
        """Save current pose as a target pose"""
        pos = self.get_current_positions()
        if pos is not None:
            self.target_poses[pose_name] = pos.copy()
            return True
        return False

    def set_target_pose(self, pose_name):
        """Set which pose to compare against"""
        if pose_name in self.target_poses:
            self.current_target = pose_name
            return True
        return False

    def calculate_pose_distance(self, target_pose):
        """Calculate euclidean distance between current and target pose"""
        current = self.get_current_positions()
        if current is None or target_pose is None:
            return None

        # Calculate per-landmark distance
        distances = np.linalg.norm(current - target_pose, axis=1)
        return distances

    def calculate_stability_score(self, n_frames=10):
        """
        Calculate how stable/still the pose is over last n frames
        Returns: 0-100 score (100 = perfectly still)
        """
        vel_hist = self.get_velocity_history(n_frames)
        if vel_hist is None or len(vel_hist) < 2:
            return 0.0

        # Average speed across all landmarks and frames
        speeds = np.linalg.norm(vel_hist, axis=2)
        avg_speed = np.mean(speeds)

        # Convert to score (lower speed = higher score)
        # Typical motion speeds are 0-5, so we normalize
        stability = max(0, 100 - (avg_speed * 20))
        return min(100, stability)

    def calculate_pose_score(self, pose_name=None,
                             position_weight=0.6,
                             stability_weight=0.4,
                             n_stability_frames=15):
        """
        Calculate overall pose score (0-100)

        Args:
            pose_name: Target pose name (uses current_target if None)
            position_weight: How much position accuracy matters (0-1)
            stability_weight: How much stability matters (0-1)
            n_stability_frames: Number of frames to check stability

        Returns:
            dict with 'total_score', 'position_score', 'stability_score', 'distances'
        """
        target_name = pose_name or self.current_target

        if target_name not in self.target_poses:
            return None

        target = self.target_poses[target_name]
        distances = self.calculate_pose_distance(target)

        if distances is None:
            return None

        # Position score: convert distances to 0-100
        # Typical good pose match has avg distance < 0.05
        avg_distance = np.mean(distances)
        position_score = max(0, 100 - (avg_distance * 1000))
        position_score = min(100, position_score)

        # Stability score
        stability_score = self.calculate_stability_score(n_stability_frames)

        # Combined score
        total_score = (position_score * position_weight +
                       stability_score * stability_weight)

        return {
            'total_score': total_score,
            'position_score': position_score,
            'stability_score': stability_score,
            'avg_distance': avg_distance,
            'distances': distances
        }

    def is_pose_locked(self, pose_name=None,
                       position_threshold=90,
                       stability_threshold=85,
                       n_frames=10):
        """
        Check if pose is "locked" (held correctly and steadily)

        Args:
            pose_name: Target pose to check
            position_threshold: Minimum position score (0-100)
            stability_threshold: Minimum stability score (0-100)
            n_frames: Must maintain for this many frames

        Returns:
            bool: True if pose is locked
        """
        scores = self.calculate_pose_score(pose_name, n_stability_frames=n_frames)

        if scores is None:
            return False

        return (scores['position_score'] >= position_threshold and
                scores['stability_score'] >= stability_threshold)

    def calculate_distance_2d(self, landmark_id_1, landmark_id_2):
        """
        Calculate 2D distance between two landmarks (ignoring depth)

        Args:
            landmark_id_1: First landmark index
            landmark_id_2: Second landmark index

        Returns:
            float: Distance in normalized coordinates (0-1 range)
        """
        positions = self.get_current_positions()
        if positions is None:
            return None

        point1 = positions[landmark_id_1]
        point2 = positions[landmark_id_2]

        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]

        distance = (dx ** 2 + dy ** 2) ** 0.5
        print(distance)
        return distance

    def get_worst_landmarks(self, pose_name=None, n=5):
        """Get the N landmarks that are furthest from target position"""
        target_name = pose_name or self.current_target

        if target_name not in self.target_poses:
            return None

        target = self.target_poses[target_name]
        distances = self.calculate_pose_distance(target)

        if distances is None:
            return None

        # Get indices of worst landmarks
        worst_indices = np.argsort(distances)[-n:][::-1]

        return [(int(idx), float(distances[idx])) for idx in worst_indices]

    # ============================================

    def process_frame(self, frame):
        """Process a single frame and update buffers"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        red_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            detector = HandTorsoDetector()
            detection = detector.is_hand_near_torso(res.pose_landmarks, threshold=0.05)

            # Convert to NumPy
            arr = np.fromiter((v for p in lm for v in (p.x, p.y, p.z)),
                              dtype=np.float32).reshape(self.NUM_LM, 3)

            self.pos_mat[self.idx] = arr

            if self.prev is not None:
                dv = (arr - self.prev) * (1.0 / self.DT)
                self.vel_mat[self.idx] = dv
                self.spd_mat[self.idx] = np.linalg.norm(dv, axis=1)
            self.prev = arr

            self.idx = (self.idx + 1) % self.N_FRAMES
            self.filled = min(self.filled + 1, self.N_FRAMES)

            # Draw visualization
            xs = arr[:, 0] * w
            ys = arr[:, 1] * h

            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())

            if self.show_bbox:
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

            if self.show_skel:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, res.pose_landmarks, self.POSE_CONNECTIONS)

            if self.show_lm:
                pts = np.column_stack((xs.astype(int), ys.astype(int)))
                for p in pts:
                    cv2.circle(frame, tuple(p), 3, (255, 0, 0), -1)
            if detection["left_hand"] or detection["right_hand"]:
                self.error_counter.count(res.pose_landmarks)
                #threading.Thread(target=playAlarm, daemon=True).start()
                pass
            """if (self.has_knife(self.get_landmark_position(23), self.get_landmark_position(19))):
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    self.POSE_CONNECTIONS,
                    landmark_drawing_spec=red_spec,
                    connection_drawing_spec=red_spec
                )"""
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
        """Release resources"""
        self.pose.close()


def run():
    logger = err.Logger()
    error_counter= err.ErrorCounter(fps=30, logger=logger)
    tracker = PoseTracker(buffer_size=30, fps=30, error_counter=error_counter)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mode = "scoring"  # or "scoring"

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

        # SCORING SYSTEM USAGE EXAMPLES:
        # Get pose scores
        scores = tracker.calculate_pose_score()


        # Display scores
        y_offset = 30
        #cv2.putText(frame, f"Mode: SCORING ({tracker.current_target})",
                   # (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        y_offset += 35
        #cv2.putText(frame, f"Total Score: {scores['total_score']:.1f}/100",
                   # (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        y_offset += 30
        #cv2.putText(frame, f"Position: {scores['position_score']:.1f}",
                   # (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)

        y_offset += 30
        #cv2.putText(frame, f"Stability: {scores['stability_score']:.1f}",
        #            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)

        # Check if pose is locked
        is_locked = tracker.is_pose_locked()
        if is_locked:
            y_offset += 35
            cv2.putText(frame, "POSE LOCKED!",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)

        # Show worst landmarks (optional)
        worst = tracker.get_worst_landmarks(n=3)
        if worst and scores['position_score'] < 90:
            y_offset += 35
            cv2.putText(frame, "Fix landmarks:",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
            for idx, dist in worst:
                y_offset += 20
                cv2.putText(frame, f"  #{idx}: {dist:.3f}",
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 255), 1)

        cv2.imshow("Pose Scoring System", frame)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('q'):
            break
        elif k == ord('s'):
            # Save current pose
            if tracker.save_target_pose("target_pose"):
                tracker.set_target_pose("target_pose")
                print("✓ Target pose saved!")
        elif k == ord('t'):
            mode = "scoring"
            print("Switched to scoring mode")
        elif k == ord('f'):
            mode = "freeform"
            print("Switched to freeform mode")
    cap.release()
    cv2.destroyAllWindows()
    tracker.release()


if __name__ == "__main__":
    run()