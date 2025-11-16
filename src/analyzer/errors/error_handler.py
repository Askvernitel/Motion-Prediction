import os
import csv
import threading
from math import trunc
import mediapipe as mp
import simpleaudio
import cv2
from datetime import datetime

#from src.analyzer.action_detector import ACT_LOOKING_OVER_LEFT_SHOULDER, ACT_LOOKING_OVER_RIGHT_SHOULDER, \
#    ACT_IN_AREA_WEAPON, ACT_LEFT_PUNCH_FORWARD, ACT_RIGHT_PUNCH_FORWARD, ACT_RIGHT_PUNCH_SIDE, ACT_LEFT_PUNCH_SIDE


ACT_LOOKING_OVER_LEFT_SHOULDER = "LOOKING OVER LEFT SHOULDER"
ACT_LOOKING_OVER_RIGHT_SHOULDER = "LOOKING OVER RIGHT SHOULDER"
ACT_LEFT_PUNCH_FORWARD = "LEFT PUNCH FORWARD"
ACT_LEFT_PUNCH_SIDE = "LEFT PUNCH SIDE"
ACT_RIGHT_PUNCH_SIDE = "RIGHT PUNCH SIDE"
ACT_RIGHT_PUNCH_FORWARD = "RIGHT PUNCH FORWARD"
ACT_IN_AREA_WEAPON = "IN AREA WEAPON"
MAX_SEC_ERROR = 3

#it is kind of frames
ERROR_SENSITIVITY_PUNCH = 1
ERROR_SENSITIVITY_LEFT_SHOULDER = 5
ERROR_SENSITIVITY_RIGHT_SHOULDER = 5
ERROR_SENSITIVITY_WEAPON_AREA = 1

lock = threading.Lock()
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

class ErrorBundle:
    level_of_error = 0
    error_type = "Unknown"
    extra_info = ""
    def __init__(self, level_of_error =0 , error_type="Unknown", extra_info=""):
        self.level_of_error = level_of_error
        self.error_type = error_type
        self.extra_info = extra_info
class LoggerInput:
    def __init__(self, error="Unknown", **kwargs):
        self.error = error
        self.custom_fields = kwargs

    def to_dict(self):
        """Convert LoggerInput to dictionary for CSV writing"""
        data = {"error": self.error}
        data.update(self.custom_fields)
        return data


class Logger:
    def __init__(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(os.getcwd(), "logs", "error_log.csv")

        self.file = file_path

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.file), exist_ok=True)

        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(self.file):
            with open(self.file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['timestamp', 'error'])
                writer.writeheader()

    def write_file(self, log_input: LoggerInput):
        """Write a LoggerInput to CSV file"""
        data = log_input.to_dict()
        self.add_timestamp(data)

        # Read existing headers
        fieldnames = ['timestamp', 'error']
        if os.path.exists(self.file):
            with open(self.file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    fieldnames = list(reader.fieldnames)

        # Add any new fields from current data
        for key in data.keys():
            if key not in fieldnames:
                fieldnames.append(key)

        # Write to CSV
        with open(self.file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(data)

    def add_timestamp(self, data_dict):
        """Add timestamp to dictionary"""
        data_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return data_dict


class ErrorCounter:
    def __init__(self, fps=30, error_type="Unknown", logger: Logger = None):
        self.fps = fps
        self.error_type = error_type
        self.logger = logger if logger else Logger()
        self.errors = 0
        self.error_frame_count = 0

    def reset_frame_counter(self):
        """Reset the error frame counter"""
        self.error_frame_count = 0
    def set_error_type(self, error_type):
        self.error_type = error_type

    def handle_error_draw(self):
        pass
    def is_error(self, pos=None):
        """Check if error threshold exceeded"""
        error_sec_amount = self.error_frame_count // self.fps
        if error_sec_amount >= MAX_SEC_ERROR:
            # Log the error
            print("POS", pos)
            self.handle_error(pos,error_sec_amount)
            return True
        return False
    def handle_error(self, pos, error_sec_amount=0, play_alarm = True):
        self.errors += 1
        log_input = LoggerInput(
            error=self.error_type,
            position=str(pos) if pos else "N/A",
            error_count=self.errors,
            duration_seconds=error_sec_amount
        )
        self.logger.write_file(log_input)
        if play_alarm:
            threading.Thread(target=playAlarm, daemon=True).start()
        print(f"ERROR: {self.error_type} exceeded {MAX_SEC_ERROR} seconds")
        self.reset_frame_counter()

    def handle_error_bundle(self, err:ErrorBundle, play_alarm = False):
        self.errors += 1
        log_input = LoggerInput(
            error=err.error_type,
            extra_info=str(err.extra_info) if err.extra_info else "N/A",
            error_count=self.errors,
            level_of_error=err.level_of_error,
        )
        self.logger.write_file(log_input)
        if play_alarm:
            threading.Thread(target=playAlarm, daemon=True).start()
        print(f"ERROR: {self.error_type} exceeded {MAX_SEC_ERROR} seconds")
        #self.reset_frame_counter()

    def count(self, pos=None, frame = None):
        """Increment error frame count and check threshold"""
        self.error_frame_count += 1
        print("ERROR FRAME COUNT", self.error_frame_count )
        print("ERROR POS", str(pos))
        return self.is_error(pos)

class ErrorService:
    def __init__(self, flag_frame_amount = 180):
        self.frame = None
        self.landmarks= None
        self.flag_frame_amount = flag_frame_amount
        self.current_flag_frame_amount = flag_frame_amount
        self.flagged = False
        self.error_counter = ErrorCounter()
    def get_error_count(self):
        pass
    def reset_flag_frame_amount(self):
        if self.current_flag_frame_amount <= 0:
            self.flagged=False
            self.current_flag_frame_amount = self.flag_frame_amount
    def set_current_frame(self, frame = None, landmarks=None):
        self.frame =  frame
        self.landmarks = landmarks
    def should_flag(self):
        return self.flagged
    def flag_frames(self):
        if self.current_flag_frame_amount <= 0:
            return
        if self.landmarks is None:
            return
        lm = self.landmarks.landmark
        frame = self.frame
        h, w = frame.shape[:2]
        size = 3  # Move this here, outside the loop
        # Get bounding box coordinates for the entire body
        x_coords = [int(landmark.x * w) for landmark in lm]
        y_coords = [int(landmark.y * h) for landmark in lm]

        # Find min/max to create bounding box
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        # Add padding if you want
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        # Draw RED rectangle around entire body
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                      (0, 0, 255), 3)  # 3 = thickness

        for i, j in mp.solutions.pose.POSE_CONNECTIONS:
            x1, y1 = int(lm[i].x * w), int(lm[i].y * h)
            x2, y2 = int(lm[j].x * w), int(lm[j].y * h)

            # Red filled rectangles at each landmark
            cv2.rectangle(frame, (x1 - size, y1 - size),
                          (x1 + size, y1 + size),
                          (0, 0, 255), -1)

            cv2.rectangle(frame, (x2 - size, y2 - size),
                          (x2 + size, y2 + size),
                          (0, 0, 255), -1)

            # Red circles (these will overlap the rectangles)
            cv2.circle(frame, (x1, y1), 3, (0, 0, 255), -1)
            cv2.circle(frame, (x2, y2), 3, (0, 0, 255), -1)

            # Red line connecting landmarks
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        self.current_flag_frame_amount -= 1
        self.reset_flag_frame_amount()
    def check_suspicious_head_movement(self, acts):
        error_bundle = ErrorBundle(error_type="SUSPICIOUS HEAD MOVEMENT")
        amount_left_head = 0
        amount_right_head = 0
        for act in acts:
            if ACT_LOOKING_OVER_LEFT_SHOULDER in act:
                amount_left_head += 1
            if ACT_LOOKING_OVER_RIGHT_SHOULDER in act:
                amount_right_head += 1
        print("####################")
        print(amount_left_head, amount_right_head)
        print("####################")
        if amount_left_head >=ERROR_SENSITIVITY_LEFT_SHOULDER and amount_right_head >=ERROR_SENSITIVITY_RIGHT_SHOULDER:
            error_bundle.level_of_error=amount_left_head + amount_right_head
            self.flagged  =True
            self.error_counter.handle_error_bundle(error_bundle, play_alarm=True)
    def check_suspicious_punch_movement(self,acts):

        error_bundle = ErrorBundle(error_type="SUSPICIOUS PUNCH MOVEMENT")

        near_weapon_call = 0
        punch_call = 0
        for act in acts:
            if ACT_IN_AREA_WEAPON in act:
                near_weapon_call += 1
            if ACT_LEFT_PUNCH_FORWARD in act or ACT_RIGHT_PUNCH_FORWARD in act or ACT_RIGHT_PUNCH_SIDE in act or ACT_LEFT_PUNCH_SIDE in act:
                punch_call += 1
        if near_weapon_call >=2 and punch_call >=2:
            #self.error_counter.handle_error_bundle(error_bundle)
            self.flagged =True
            error_bundle.level_of_error=punch_call+near_weapon_call
            self.error_counter.handle_error_bundle(error_bundle, play_alarm=True)

    def check_error(self, act_history):
        self.check_suspicious_head_movement(act_history)
        self.check_suspicious_punch_movement(act_history)

#INPUTIS MAGALITI AQ ARIS LOGGERISTVIS
"""
    INPUTIS MAGALIT ErrorCounter Ar Aris Aucilebeli
    # Create logger and error counter
    logger = Logger()
    counter = ErrorCounter(fps=30, error_type="ConnectionTimeout", logger=logger)

    # Simulate some errors
    for i in range(100):
        counter.count(pos=f"frame_{i}")

    # You can also manually log events
    manual_log = LoggerInput(
        error="CustomError",
        severity="HIGH",
        message="Something went wrong",
        user_id="12345"
    )
    logger.write_file(manual_log)

    print(f"Logs written to: {logger.file}")
"""