import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose Detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize pose detector
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1  # 0=lite, 1=full, 2=heavy
)

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Human Body Detection Started. Press 'q' to quit.")
print("Controls:")
print("  's' - Toggle skeleton overlay")
print("  'b' - Toggle bounding box")
print("  'l' - Toggle landmark points")

# Display options
show_skeleton = True
show_bbox = True
show_landmarks = True

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Flip frame horizontally for selfie-view
    frame = cv2.flip(frame, 1)
    
    # Get image dimensions
    h, w, c = frame.shape
    
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame for pose detection
    results = pose.process(rgb_frame)
    
    # Draw pose landmarks if detected
    if results.pose_landmarks:
        # Extract landmark coordinates
        landmarks = results.pose_landmarks.landmark
        
        # Calculate bounding box around the person
        x_coords = [lm.x * w for lm in landmarks]
        y_coords = [lm.y * h for lm in landmarks]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add padding to bounding box
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Draw bounding box
        if show_bbox:
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, "HUMAN DETECTED", (x_min, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw skeleton connections
        if show_skeleton:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw individual landmark points
        if show_landmarks:
            for idx, landmark in enumerate(landmarks):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
        
        # Display key body part positions
        # Nose
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        nose_x, nose_y = int(nose.x * w), int(nose.y * h)
        
        # Left and right shoulders
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        # Calculate if person is centered
        center_x = w // 2
        person_center = (x_min + x_max) // 2
        
        if abs(person_center - center_x) < 50:
            position_text = "Centered"
            pos_color = (0, 255, 0)
        elif person_center < center_x:
            position_text = "Left Side"
            pos_color = (0, 165, 255)
        else:
            position_text = "Right Side"
            pos_color = (0, 165, 255)
        
        cv2.putText(frame, f"Position: {position_text}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, pos_color, 2)
        
        # Estimate pose (standing, arms up, etc.)
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
            pose_text = "Arms Raised"
        else:
            pose_text = "Normal Pose"
        
        cv2.putText(frame, f"Pose: {pose_text}", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Status indicator
        cv2.putText(frame, "HUMAN DETECTED", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Human Detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display instructions
    cv2.putText(frame, "Press 's/b/l to toggle | 'q' to quit", (10, h - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show the frame
    cv2.imshow("Human Body Detection - MediaPipe", frame)
    
    # Handle key presses
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        show_skeleton = not show_skeleton
        print(f"Skeleton overlay: {'ON' if show_skeleton else 'OFF'}")
    elif key == ord('b'):
        show_bbox = not show_bbox
        print(f"Bounding box: {'ON' if show_bbox else 'OFF'}")
    elif key == ord('l'):
        show_landmarks = not show_landmarks
        print(f"Landmark points: {'ON' if show_landmarks else 'OFF'}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
pose.close()
