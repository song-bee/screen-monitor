import cv2
import numpy as np
from PIL import ImageGrab
import time

# Configuration Constants
MOTION_THRESHOLD = 50000  # Motion sensitivity threshold
COLOR_VARIANCE_THRESHOLD = 2000  # Adjust this based on screen content
DETECTION_TIME = 30  # Seconds required to confirm video/game
FRAME_RATE = 2  # Frames per second for capturing screen
IGNORE_LOW_MOTION_TIME = 2  # Tolerated low-motion duration (seconds)
RESET_TIME = 3  # Continuous low-motion duration before reset (seconds)

# State Variables
motion_start_time = None
low_motion_start_time = None
prev_frame = None

def capture_screen():
    """Captures a screenshot and converts it to numpy array."""
    screenshot = ImageGrab.grab()
    return np.array(screenshot)

def calculate_motion(prev_frame, current_frame):
    """Computes motion intensity by comparing two frames."""
    if prev_frame is None:
        return 0  # No motion on the first frame
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    gray_current = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(gray_prev, gray_current)
    return np.sum(diff)  # Sum of pixel differences (motion level)

def analyze_color_richness(frame):
    """Analyzes color richness using variance of saturation and brightness."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    saturation = hsv_frame[:, :, 1]  # Extract saturation channel
    value = hsv_frame[:, :, 2]  # Extract brightness channel

    sat_variance = np.var(saturation)  # Color intensity variation
    val_variance = np.var(value)  # Brightness variation

    color_richness = (sat_variance + val_variance) / 2
    return color_richness

def update_motion_state(motion_level, color_richness):
    """
    Tracks motion duration and determines if a game/video is detected.
    Considers both motion and color richness.
    """
    global motion_start_time, low_motion_start_time

    if motion_level > MOTION_THRESHOLD and color_richness > COLOR_VARIANCE_THRESHOLD:
        # If significant motion and rich color are detected
        if motion_start_time is None:
            motion_start_time = time.time()  # Start motion timer
            low_motion_start_time = None  # Reset low motion timer

        elapsed_time = time.time() - motion_start_time
        if elapsed_time >= DETECTION_TIME:
            print("🎮 Video or Game Detected (Motion + Color)!")
            motion_start_time = None  # Prevent repeated alerts

    else:
        # Motion is below the threshold
        if motion_start_time is not None:
            if low_motion_start_time is None:
                low_motion_start_time = time.time()  # Start low motion timer
            
            low_elapsed_time = time.time() - low_motion_start_time
            if low_elapsed_time > RESET_TIME:
                print("❌ Motion Reset (No activity for too long)")
                motion_start_time = None  # Reset detection state

def detect_long_motion():
    """Main function that continuously monitors the screen for sustained motion & color analysis."""
    global prev_frame
    while True:
        current_frame = capture_screen()
        motion_level = calculate_motion(prev_frame, current_frame)
        color_richness = analyze_color_richness(current_frame)

        print(f"Motion Level: {motion_level} | Color Richness: {color_richness}")  # Debugging output

        update_motion_state(motion_level, color_richness)

        prev_frame = current_frame  # Store the last frame
        time.sleep(1 / FRAME_RATE)  # Adjust capture rate

# Run the detection
detect_long_motion()
