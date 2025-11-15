import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import json
from datetime import datetime
import pickle

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
# Camera selection: 0 = built-in webcam, 1 = first USB webcam, 2 = second USB webcam, etc.
CAMERA_INDEX = 0 # Change this to switch cameras (0, 1, 2, ...)

cap = cv2.VideoCapture(CAMERA_INDEX)

# Set resolution to 1080p
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1270)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Verify the resolution was set
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

if not cap.isOpened():
    print(f"Error: Could not open camera {CAMERA_INDEX}")
    print("Available cameras to try: 0 (built-in), 1 (USB), 2 (second USB)...")
    exit(1)

print(f"Using camera index: {CAMERA_INDEX}")
print(f"Resolution set to: {int(actual_width)}x{int(actual_height)}")

print("Human Body Detection Started. Press 'q' to quit.")
print("Controls:")
print("  's' - Toggle skeleton overlay")
print("  'b' - Toggle bounding box")
print("  'l' - Toggle landmark points")
print("  'v' - Toggle velocity vectors display")
print("  't' - Toggle trajectory trails")
print("  'SPACE' - Save movement vector matrix to file")
print("  'r' - Start/stop continuous recording")
print("  '+/-' - Increase/decrease frames to track")

# Display options
show_skeleton = True
show_bbox = True
show_landmarks = True
show_vectors = True  # Changed default to True
show_trails = False

# Recording state
is_recording = False
recording_start_time = None
recorded_chunks = []  # Store multiple chunks of data

# Movement tracking parameters
N_FRAMES = 30  # Number of frames to track (adjustable)
NUM_LANDMARKS = 33  # MediaPipe has 33 pose landmarks
VECTOR_DIM = 3  # x, y, z velocity components

# Initialize the 3D matrix: [frames, landmarks, vector_components]
# Shape: (N_FRAMES, 33, 3) for velocity data (vx, vy, vz)
movement_matrix = np.zeros((N_FRAMES, NUM_LANDMARKS, VECTOR_DIM), dtype=np.float32)

# Additional matrices for position and speed
position_matrix = np.zeros((N_FRAMES, NUM_LANDMARKS, 3), dtype=np.float32)  # x, y, z positions
speed_matrix = np.zeros((N_FRAMES, NUM_LANDMARKS), dtype=np.float32)  # scalar speed

# Circular buffer index
current_frame_idx = 0
frames_filled = 0

# Previous frame landmarks for velocity calculation
prev_landmarks = None
frame_count = 0

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
        
        # Convert landmarks to numpy array for easier manipulation
        current_landmarks = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
        for idx, lm in enumerate(landmarks):
            current_landmarks[idx] = [lm.x, lm.y, lm.z]
        
        # Store position in matrix
        position_matrix[current_frame_idx] = current_landmarks
        
        # Calculate velocities if we have a previous frame
        velocities = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
        if prev_landmarks is not None:
            # Assume 30 fps for time calculation
            time_diff = 1.0 / 30.0
            
            # Calculate velocity for each landmark
            velocities = (current_landmarks - prev_landmarks) / time_diff
            
            # Store velocity in matrix
            movement_matrix[current_frame_idx] = velocities
            
            # Calculate and store speed (magnitude of velocity)
            speeds = np.linalg.norm(velocities, axis=1)
            speed_matrix[current_frame_idx] = speeds
        
        # Update previous landmarks for next frame
        prev_landmarks = current_landmarks.copy()
        
        # Update circular buffer index
        current_frame_idx = (current_frame_idx + 1) % N_FRAMES
        frames_filled = min(frames_filled + 1, N_FRAMES)
        
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
        
        # Draw trajectory trails showing movement over past frames
        if show_trails and frames_filled > 1:
            # Draw trails for key landmarks (e.g., wrists, ankles)
            key_landmarks = [15, 16, 27, 28]  # Left wrist, right wrist, left ankle, right ankle
            trail_colors = [(0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0)]
            
            for trail_idx, landmark_idx in enumerate(key_landmarks):
                color = trail_colors[trail_idx % len(trail_colors)]
                
                # Draw line connecting past positions
                points = []
                for i in range(max(0, frames_filled - 15), frames_filled):  # Last 15 frames
                    frame_idx = i % N_FRAMES
                    pos = position_matrix[frame_idx, landmark_idx]
                    px = int(pos[0] * w)
                    py = int(pos[1] * h)
                    points.append((px, py))
                
                # Draw the trail
                for i in range(len(points) - 1):
                    alpha = (i + 1) / len(points)  # Fade effect
                    thickness = max(1, int(3 * alpha))
                    cv2.line(frame, points[i], points[i + 1], color, thickness)
        
        # Draw velocity vectors as lines
        if show_vectors and prev_landmarks is not None:
            # Get the most recent velocity data
            prev_idx = (current_frame_idx - 1) % N_FRAMES
            velocities_to_draw = movement_matrix[prev_idx]
            
            for idx, landmark in enumerate(landmarks):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                vx, vy, vz = velocities_to_draw[idx]
                
                # Scale velocity for visualization (adjustable)
                scale = 500  # Reduced scale for better visibility
                vx_scaled = int(vx * scale)
                vy_scaled = int(vy * scale)
                
                # Only draw if there's significant movement
                if abs(vx_scaled) > 2 or abs(vy_scaled) > 2:
                    end_x = x + vx_scaled
                    end_y = y + vy_scaled
                    
                    # Get speed for this landmark
                    speed = speed_matrix[prev_idx, idx]
                    
                    # Color based on speed (green=slow, yellow=medium, red=fast)
                    speed_normalized = min(speed * 50, 1.0)
                    
                    if speed_normalized < 0.3:
                        color = (0, 255, 0)  # Green - slow
                    elif speed_normalized < 0.7:
                        color = (0, 255, 255)  # Yellow - medium
                    else:
                        color = (0, 0, 255)  # Red - fast
                    
                    # Draw line with arrow
                    cv2.arrowedLine(frame, (x, y), (end_x, end_y), color, 2, tipLength=0.2)
                    
                    # Optionally draw speed value next to fast-moving landmarks
                    if speed > 0.01:
                        cv2.putText(frame, f"{speed:.2f}", (end_x + 5, end_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
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
        
        # Display movement statistics
        if frames_filled > 0:
            # Calculate statistics from the matrix
            # Get valid frames (those that have been filled)
            valid_frames = min(frames_filled, N_FRAMES)
            valid_speeds = speed_matrix[:valid_frames]
            
            avg_speed = np.mean(valid_speeds)
            max_speed = np.max(valid_speeds)
            
            cv2.putText(frame, f"Avg Speed: {avg_speed:.3f}", (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Max Speed: {max_speed:.3f}", (10, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Frames: {frames_filled}/{N_FRAMES}", (10, 190),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Recording indicator
            if is_recording:
                elapsed = (datetime.now() - recording_start_time).total_seconds()
                cv2.putText(frame, f"REC {elapsed:.1f}s", (w - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                # Red dot
                cv2.circle(frame, (w - 170, 25), 8, (0, 0, 255), -1)
            
            cv2.putText(frame, f"Matrix: ({N_FRAMES}, {NUM_LANDMARKS}, {VECTOR_DIM})", (10, 220),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Status indicator
        cv2.putText(frame, "HUMAN DETECTED", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Human Detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display instructions
    cv2.putText(frame, "SPACE:save | r:record | t:trails | +/-:frames | s/b/l/v:toggle | q:quit", (10, h - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
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
    elif key == ord('v'):
        show_vectors = not show_vectors
        print(f"Velocity vectors: {'ON' if show_vectors else 'OFF'}")
    elif key == ord('t'):
        show_trails = not show_trails
        print(f"Trajectory trails: {'ON' if show_trails else 'OFF'}")
    elif key == ord('r'):
        # Toggle recording
        is_recording = not is_recording
        if is_recording:
            recording_start_time = datetime.now()
            print(f"\n🔴 Recording started at {recording_start_time.strftime('%H:%M:%S')}")
        else:
            # Save the current buffer when stopping
            if frames_filled > 0:
                valid_frames = min(frames_filled, N_FRAMES)
                chunk = {
                    'velocity_matrix': movement_matrix[:valid_frames].copy(),
                    'position_matrix': position_matrix[:valid_frames].copy(),
                    'speed_matrix': speed_matrix[:valid_frames].copy(),
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'duration': valid_frames / 30.0
                }
                recorded_chunks.append(chunk)
                print(f"⏹️  Recording stopped. Chunk saved ({valid_frames} frames, {valid_frames/30.0:.2f}s)")
                print(f"   Total chunks recorded: {len(recorded_chunks)}")
            else:
                print("⏹️  Recording stopped (no data captured)")
    elif key == ord('+') or key == ord('='):
        # Increase frame tracking
        N_FRAMES = min(N_FRAMES + 10, 300)  # Max 300 frames (10 seconds at 30fps)
        # Resize matrices
        old_movement = movement_matrix.copy()
        old_position = position_matrix.copy()
        old_speed = speed_matrix.copy()
        
        movement_matrix = np.zeros((N_FRAMES, NUM_LANDMARKS, VECTOR_DIM), dtype=np.float32)
        position_matrix = np.zeros((N_FRAMES, NUM_LANDMARKS, 3), dtype=np.float32)
        speed_matrix = np.zeros((N_FRAMES, NUM_LANDMARKS), dtype=np.float32)
        
        # Copy old data
        copy_size = min(frames_filled, N_FRAMES)
        movement_matrix[:copy_size] = old_movement[:copy_size]
        position_matrix[:copy_size] = old_position[:copy_size]
        speed_matrix[:copy_size] = old_speed[:copy_size]
        
        print(f"Frame tracking increased to {N_FRAMES} frames ({N_FRAMES/30.0:.1f}s)")
    elif key == ord('-') or key == ord('_'):
        # Decrease frame tracking
        N_FRAMES = max(N_FRAMES - 10, 10)  # Min 10 frames
        # Resize matrices
        old_movement = movement_matrix.copy()
        old_position = position_matrix.copy()
        old_speed = speed_matrix.copy()
        
        movement_matrix = np.zeros((N_FRAMES, NUM_LANDMARKS, VECTOR_DIM), dtype=np.float32)
        position_matrix = np.zeros((N_FRAMES, NUM_LANDMARKS, 3), dtype=np.float32)
        speed_matrix = np.zeros((N_FRAMES, NUM_LANDMARKS), dtype=np.float32)
        
        # Copy old data (truncated if necessary)
        copy_size = min(frames_filled, N_FRAMES)
        movement_matrix[:copy_size] = old_movement[:copy_size]
        position_matrix[:copy_size] = old_position[:copy_size]
        speed_matrix[:copy_size] = old_speed[:copy_size]
        
        frames_filled = min(frames_filled, N_FRAMES)
        current_frame_idx = current_frame_idx % N_FRAMES
        
        print(f"Frame tracking decreased to {N_FRAMES} frames ({N_FRAMES/30.0:.1f}s)")
    elif key == ord(' '):  # Spacebar
        # Save movement matrix data (current buffer + all recorded chunks)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Combine current buffer with recorded chunks if any
        all_data = []
        
        # Add previously recorded chunks
        if len(recorded_chunks) > 0:
            print(f"\n💾 Saving {len(recorded_chunks)} recorded chunks...")
            for i, chunk in enumerate(recorded_chunks):
                all_data.append({
                    'chunk_id': i,
                    'velocity_matrix': chunk['velocity_matrix'],
                    'position_matrix': chunk['position_matrix'],
                    'speed_matrix': chunk['speed_matrix'],
                    'timestamp': chunk['timestamp'],
                    'duration': chunk['duration']
                })
        
        # Add current buffer
        if frames_filled > 0:
            valid_frames = min(frames_filled, N_FRAMES)
            all_data.append({
                'chunk_id': len(recorded_chunks),
                'velocity_matrix': movement_matrix[:valid_frames].copy(),
                'position_matrix': position_matrix[:valid_frames].copy(),
                'speed_matrix': speed_matrix[:valid_frames].copy(),
                'timestamp': timestamp,
                'duration': valid_frames / 30.0
            })
            
            # Save as numpy file (.npz) - efficient for matrices
            filename_npz = f"movement_recording_{timestamp}.npz"
            
            # Prepare data for saving
            save_dict = {
                'num_chunks': len(all_data),
                'metadata': {
                    'total_chunks': len(all_data),
                    'num_landmarks': NUM_LANDMARKS,
                    'vector_dim': VECTOR_DIM,
                    'fps': 30,
                    'timestamp': timestamp,
                }
            }
            
            # Add each chunk to the save dictionary
            for chunk in all_data:
                chunk_id = chunk['chunk_id']
                save_dict[f'chunk_{chunk_id}_velocity'] = chunk['velocity_matrix']
                save_dict[f'chunk_{chunk_id}_position'] = chunk['position_matrix']
                save_dict[f'chunk_{chunk_id}_speed'] = chunk['speed_matrix']
                save_dict[f'chunk_{chunk_id}_metadata'] = {
                    'timestamp': chunk['timestamp'],
                    'duration': chunk['duration'],
                    'frames': len(chunk['velocity_matrix'])
                }
            
            np.savez(filename_npz, **save_dict)
            
            print(f"\n✓ Saved movement recording to {filename_npz}")
            print(f"  Total chunks: {len(all_data)}")
            total_duration = sum([chunk['duration'] for chunk in all_data])
            print(f"  Total duration: {total_duration:.2f}s")
            print(f"  Matrix shape per chunk: (frames, {NUM_LANDMARKS}, {VECTOR_DIM})")
            
            # Also save as pickle
            filename_pkl = f"movement_recording_{timestamp}.pkl"
            with open(filename_pkl, 'wb') as f:
                pickle.dump({'chunks': all_data, 'metadata': save_dict['metadata']}, f)
            print(f"✓ Also saved as {filename_pkl}")
            
            # Clear recorded chunks after saving
            recorded_chunks.clear()
            print(f"✓ Cleared recorded chunks from memory\n")
        else:
            print("✗ No movement data to save yet")
    
    # Auto-save during recording every 100 frames
    if is_recording and frames_filled > 0 and frame_count % 100 == 0:
        valid_frames = min(frames_filled, N_FRAMES)
        chunk = {
            'velocity_matrix': movement_matrix[:valid_frames].copy(),
            'position_matrix': position_matrix[:valid_frames].copy(),
            'speed_matrix': speed_matrix[:valid_frames].copy(),
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'duration': valid_frames / 30.0
        }
        recorded_chunks.append(chunk)
        print(f"📦 Auto-saved chunk {len(recorded_chunks)} ({valid_frames} frames)")
    
    frame_count += 1

# Cleanup
cap.release()
cv2.destroyAllWindows()
pose.close()
