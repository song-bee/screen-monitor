import cv2
import math
import numpy as np
from PIL import ImageGrab
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import platform

if platform.system() == 'Darwin':
    import AppKit
    import objc
    from PyObjCTools.AppHelper import runConsoleEventLoop

class StatusBarApp:
    """Creates a Mac status bar item to display motion & color richness."""
    
    def __init__(self):
        self.app = AppKit.NSApplication.sharedApplication()
        self.status_bar = AppKit.NSStatusBar.systemStatusBar()
        self.status_item = self.status_bar.statusItemWithLength_(AppKit.NSVariableStatusItemLength)
        self.status_item.setHighlightMode_(True)
        self.status_item.button().setTitle_("Loading...")

    def update_status(self, motion_value, color_value, elapsed_time):
        """Updates the status bar text dynamically."""
        motion_text = f"{motion_value:.2f}"  # Format to 2 decimal places
        color_text = f"{color_value:.2f}"
        elapsed_time_text = f"{int(elapsed_time)}"
        self.status_item.button().setTitle_(f"M: {motion_text} | C: {color_text} | E: {elapsed_time_text}")

# Configuration Constants
MOTION_LOG_THRESHOLD = 6 # Motion sensitivity threshold
COLOR_LOG_THRESHOLD = 3 # Adjust based on screen content
DETECTION_TIME = 30  # Seconds required to confirm video/game
FRAME_RATE = 2  # Frames per second for capturing screen
IGNORE_LOW_MOTION_TIME = 2  # Tolerated low-motion duration (seconds)
RESET_TIME = 3  # Continuous low-motion duration before reset (seconds)

# State Variables
motion_start_time = None
low_motion_start_time = None
prev_frame = None

# Data storage for past 30 seconds (ensures all deques have same length)
max_data_points = DETECTION_TIME * FRAME_RATE  # Store last 30 sec of data
motion_data = deque([0] * max_data_points, maxlen=max_data_points)
color_data = deque([0] * max_data_points, maxlen=max_data_points)
time_labels = deque(np.linspace(-DETECTION_TIME, 0, max_data_points), maxlen=max_data_points)

def capture_screen():
    """Captures a screenshot, excluding the macOS status bar and menus."""
    screenshot = ImageGrab.grab()
    screen_array = np.array(screenshot)

    # Get screen size
    screen_height, screen_width, _ = screen_array.shape

    # Define crop boundaries (Adjust these based on macOS screen layout)
    menu_bar_height = 200  # Approximate height of macOS status bar & menu bar

    # Crop only the main screen area
    cropped_screen = screen_array[menu_bar_height:screen_height - 100, 100:screen_width - 100]

    return cropped_screen

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

def update_motion_state(motion_log, color_log):
    """
    Tracks motion duration and determines if a game/video is detected.
    Uses log10 values and ignores temporary fluctuations under 3 seconds.
    """
    global motion_start_time, low_motion_start_time

    elapsed_time = 0

    if motion_log > MOTION_LOG_THRESHOLD and color_log > COLOR_LOG_THRESHOLD:
        # Significant motion and color richness detected
        if motion_start_time is None:
            motion_start_time = time.time()  # Start motion timer
            low_motion_start_time = None  # Reset low motion timer

        elapsed_time = time.time() - motion_start_time

        if elapsed_time >= DETECTION_TIME:
            print("🎮 Video or Game Detected (Motion + Color)!")
            motion_start_time = None  # Prevent repeated alerts

    else:
        # Motion/color richness dropped below threshold
        if motion_start_time is not None:
            if low_motion_start_time is None:
                low_motion_start_time = time.time()  # Start low motion timer
            
            low_elapsed_time = time.time() - low_motion_start_time

            if low_elapsed_time > 3:  # Ignore short fluctuations (< 3 sec)
                print("❌ Motion Reset (No activity for too long)")
                motion_start_time = None  # Reset detection state
                low_motion_start_time = None  # Reset low-motion tracking

    return elapsed_time

def update_plot(i):
    """Updates the live plot with new motion and color richness values."""
    global prev_frame

    current_frame = capture_screen()
    motion_level = calculate_motion(prev_frame, current_frame)
    color_richness = analyze_color_richness(current_frame)

    print(f"Motion Level: {motion_level} | Color Richness: {color_richness}")  # Debugging output

    # Update the data history while keeping all deques the same length
    motion_log = math.log10(max(1, abs(motion_level)))
    color_log = math.log10(max(1, abs(color_richness)))

    motion_data.append(motion_log)
    color_data.append(color_log)

    elapsed_time = update_motion_state(motion_log, color_log)
    
    if platform.system() == 'Darwin':
        status_app.update_status(motion_log, color_log, elapsed_time)  # Update Mac status bar

    time_labels.append(time_labels[-1] + (1 / FRAME_RATE))  # Maintain correct time scale

    # Ensure time_labels matches the data size
    while len(time_labels) > max_data_points:
        time_labels.popleft()
    while len(motion_data) > max_data_points:
        motion_data.popleft()
    while len(color_data) > max_data_points:
        color_data.popleft()

    # Clear and update plots
    ax.clear()
    ax.plot(time_labels, motion_data, label="Motion Level", color="blue")
    ax.plot(time_labels, color_data, label="Color Richness", color="red")
    
    # Formatting
    ax.set_xlim(time_labels[0], time_labels[-1])
    ax.set_ylim(0, max(max(motion_data), max(color_data)) * 1.2)  # Dynamic scaling
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Value")
    ax.legend()
    ax.set_title("Live Motion & Color Analysis")

    prev_frame = current_frame  # Store the last frame

if platform.system() == 'Darwin':
    status_app = StatusBarApp()  # Initialize the status bar

# Initialize Matplotlib Figure
fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig, update_plot, interval=int(1000 / FRAME_RATE), save_count=30)

# Show the live plot
plt.show()

if platform.system() == 'Darwin':
    runConsoleEventLoop()
