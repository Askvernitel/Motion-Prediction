import cv2
import mediapipe as mp

from pose_tracker import PoseTracker
from pose_tracker import HandTorsoDetector
from pose_tracker import (
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_SHOULDER, RIGHT_SHOULDER
)
import errors.error_handler as err


mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
FPS = 30
HISTORY_SEC = 10

ACT_LOOKING_OVER_LEFT_SHOULDER = "LOOKING OVER LEFT SHOULDER"
ACT_LOOKING_OVER_RIGHT_SHOULDER = "LOOKING OVER RIGHT SHOULDER"
ACT_LEFT_PUNCH_FORWARD = "LEFT PUNCH FORWARD"
ACT_LEFT_PUNCH_SIDE = "LEFT PUNCH SIDE"
ACT_RIGHT_PUNCH_SIDE = "RIGHT PUNCH SIDE"
ACT_RIGHT_PUNCH_FORWARD = "RIGHT PUNCH FORWARD"
ACT_IN_AREA_WEAPON = "IN AREA WEAPON"
def draw_landmarks(frame, landmarks):
    """Draws pose skeleton + landmarks on the frame."""
    if landmarks is None:
        return

    mp_draw.draw_landmarks(
        frame,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_draw.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_draw.DrawingSpec(
            color=(255, 0, 0), thickness=2)
    )


def draw_body_box(frame, tracker):
    """Draw bounding box around the whole body."""
    pos = tracker.get_current_positions()
    if pos is None:
        return

    h, w = frame.shape[:2]
    xs = pos[:, 0] * w
    ys = pos[:, 1] * h

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)


def main():
    logger = err.Logger()
    error_counter = err.ErrorCounter(fps=FPS, logger=logger)
    error_service = err.ErrorService()
    tracker = PoseTracker(buffer_size=30, fps=FPS)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    action_history_secs = HISTORY_SEC
    action_history_fps = FPS
    print("=== ACTION DETECTOR WITH DISPLAY ===")
    print("Press 'q' to quit.")

    actions_history = [[] for i in range(0, action_history_fps*action_history_secs)]
    currentHistIdx = 0
    idxMod=action_history_fps*action_history_secs
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)

        # Process frame for pose data
        landmarks = tracker.process_frame_no_draw(frame)

        # Draw pose
        draw_landmarks(frame, landmarks)

        # Bounding box
        draw_body_box(frame, tracker)

        # -----------------------------------------------------------
        # ACTION DETECTION
        # -----------------------------------------------------------
        actions = []
        detector = HandTorsoDetector()
        detection = detector.is_hand_near_torso(landmarks, threshold=0.1)

        left_punch = tracker.is_punching(LEFT_WRIST, LEFT_ELBOW, LEFT_SHOULDER)
        right_punch = tracker.is_punching(RIGHT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER)

        if left_punch:
            if tracker.is_punch_toward_forward(LEFT_WRIST, LEFT_ELBOW):
                actions.append(ACT_LEFT_PUNCH_FORWARD)
            else:
                actions.append(ACT_LEFT_PUNCH_SIDE)
        if detection["left_hand"] or detection["right_hand"]:
          #  error_counter.set_error_type("Suspicious Body Movement Near Torso")
           # error_counter.count(landmarks)
            actions.append(ACT_IN_AREA_WEAPON)
        if right_punch:
            if tracker.is_punch_toward_forward(RIGHT_WRIST, RIGHT_ELBOW):
                actions.append(ACT_RIGHT_PUNCH_FORWARD)
            else:
                actions.append(ACT_RIGHT_PUNCH_SIDE)

        shoulder_side = tracker.is_checking_over_shoulder()
        if shoulder_side == "left":
            actions.append(ACT_LOOKING_OVER_LEFT_SHOULDER)
        elif shoulder_side == "right":
            actions.append(ACT_LOOKING_OVER_RIGHT_SHOULDER)

        # -----------------------------------------------------------
        # PRINT + DISPLAY ACTIONS ON SCREEN
        # -----------------------------------------------------------
        if actions:
            print(" | ".join(actions))
        error_service.set_current_frame(frame,landmarks)
        print("SHULD FLAG:", error_service.should_flag())
        if error_service.should_flag():
            error_service.flag_frames()

        y = 30
        for act in actions:
            cv2.putText(frame, act, (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 255), 2)
            y += 40

        actions_history[currentHistIdx]=actions
        currentHistIdx = (currentHistIdx + 1)%action_history_secs
        error_service.check_error(actions_history)
        # -----------------------------------------------------------
        # SHOW WINDOW
        # -----------------------------------------------------------
        cv2.imshow("Action Detector", frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.release()


if __name__ == "__main__":
    main()
