import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime


class PoseTracker:
    def __init__(self, buffer_size=30, fps=30):
        self.mp_pose = mp.solutions.pose
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

    def process_frame(self, frame):
        """Process a single frame and update buffers"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

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
    tracker = PoseTracker(buffer_size=30, fps=30)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        frame = tracker.process_frame(frame)

        # INTERFACE USAGE EXAMPLES:

        # Get current positions
        pos = tracker.get_current_positions()
        if pos is not None:
            # pos shape: (33, 3) - all landmarks [x, y, z]
            pass

        # Get current velocities
        vel = tracker.get_current_velocities()
        if vel is not None:
            # vel shape: (33, 3) - all landmark velocities [vx, vy, vz]
            pass

        # Get specific landmark (e.g., right wrist = 16)
        wrist_pos = tracker.get_landmark_position(16)
        wrist_vel = tracker.get_landmark_velocity(16)

        # Get average speed
        avg_spd = tracker.get_average_speed()

        # Get fastest moving landmark
        fastest = tracker.get_max_speed_landmark()

        # Get history (last 10 frames)
        pos_hist = tracker.get_position_history(10)

        # Display info
        if pos is not None:
            cv2.putText(frame, f"Avg Speed: {avg_spd:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("pose", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.release()