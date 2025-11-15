import os
import csv
import threading
import simpleaudio
from datetime import datetime
MAX_SEC_ERROR = 3

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
            self.errors += 1
            log_input = LoggerInput(
                error=self.error_type,
                position=str(pos) if pos else "N/A",
                error_count=self.errors,
                duration_seconds=error_sec_amount
            )
            self.logger.write_file(log_input)
            threading.Thread(target=playAlarm, daemon=True).start()
            print(f"ERROR: {self.error_type} exceeded {MAX_SEC_ERROR} seconds")
            self.reset_frame_counter()
            return True
        return False

    def count(self, pos=None, frame = None):
        """Increment error frame count and check threshold"""
        self.error_frame_count += 1
        print("ERROR FRAME COUNT", self.error_frame_count )
        print("ERROR POS", str(pos))
        return self.is_error(pos)


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