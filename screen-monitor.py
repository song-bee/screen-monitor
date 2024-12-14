from PIL import Image
from datetime import datetime
from plyer import notification

import os
import platform
import pyautogui
import pytesseract
import subprocess
import threading
import time

MAX_NOT_ALLOWED_TIME = 3
MAX_NOTIFICATION_TIMES = 3
LOCK_INTERVAL_SECONDS = 10

class ScreenMonitor:
    def __init__(self):
        self.screenshot_dir = "screenshots"
        self.last_text = ""
        self.last_text_time = datetime.now()
        self.text_stable = True
        self.warning_count = 0
        self.lock_timer_start = None
        
        # Create screenshots directory if it doesn't exist
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)

    def take_screenshot(self):
        """Take a screenshot and save it to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.screenshot_dir}/screenshot_{timestamp}.png"
        
        # Take screenshot using pyautogui
        try:
            screenshot = pyautogui.screenshot()
            screenshot.save(filename)
            return filename
        except Exception as e:
            print(f"Error screenshot: {e}")
            return None

 
    def extract_text(self, image_path):
        """Extract text from image using tesseract"""
        try:
            # Use tesseract to extract text
            text = pytesseract.image_to_string(Image.open(image_path))
            # Clean up the text
            text = text.strip()
            return text
        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""

    def is_allowed(self, content_text):
        """Check if the content is allowed"""
        if 'chrome-extension' in content_text:
            return False
        
        if 'chrome' in content_text and 'extension' in content_text:
            return False
        
        return True  # Return True if none of the disallowed conditions are met

    def notify(self, title, subtitle, message):
        system = platform.system()

        if system == 'Darwin':
            t = '-title {!r}'.format(title)
            s = '-subtitle {!r}'.format(subtitle)
            m = '-message {!r}'.format(message)
            os.system('terminal-notifier {}'.format(' '.join([m, t, s])))
        elif system == 'Linux':
            pass
        else:
            pass

    def cleanup_screenshots(self):
        """Delete old screenshot files"""
        for file in os.listdir(self.screenshot_dir):
            file_path = os.path.join(self.screenshot_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

    def lock_screen(self):
        """Lock the screen based on the operating system"""
        system = platform.system()
        
        try:
            if system == "Darwin":  # macOS
                subprocess.run(["pmset", "displaysleepnow"])
            elif system == "Windows":
                subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"])
            elif system == "Linux":
                subprocess.run(["xdg-screensaver", "lock"])
            print("Screen locked")
        except Exception as e:
            print(f"Error locking screen: {e}")

    def show_final_warning(self):
        warning_msg = "Final warning: Screen will be locked in 30 seconds"

        title = "Screen Lock Warning"
        
        self.notify(title,
            "",
            warning_msg)

    def show_warning(self):
        """Show warning notification and increment warning count"""
        self.warning_count += 1
        warning_msg = (
            f"Screen will be locked in 30 seconds after {MAX_NOTIFICATION_TIMES} warnings" 
            if self.warning_count < MAX_NOTIFICATION_TIMES
            else "Final warning: Screen will be locked in 30 seconds"
        )

        title = f"Screen Lock Warning ({self.warning_count}/{MAX_NOTIFICATION_TIMES})"
        
        '''
        notification.notify(
            title=title
            message=warning_msg,
            timeout=10
        )
        '''

        self.notify(title,
            "",
            warning_msg)

    def monitor(self):
        """Main monitoring loop"""
        try:
            not_allowed_count = 0
            lock_warning_shown = False
            final_lock_timer_start = None
            
            while True:
                # Wait for 1 second
                time.sleep(1)

                print(not_allowed_count, lock_warning_shown, self.warning_count, self.lock_timer_start, final_lock_timer_start)

                if final_lock_timer_start and time.time() - final_lock_timer_start >= LOCK_INTERVAL_SECONDS:
                    self.lock_screen()

                    # Reset everything after locking
                    not_allowed_count = 0
                    lock_warning_shown = False
                    self.warning_count = 0
                    self.lock_timer_start = None
                    final_lock_timer_start = None

                    continue
        
                # Take screenshot
                screenshot_path = self.take_screenshot()

                if not screenshot_path:
                    continue
                
                # Extract text
                current_text = self.extract_text(screenshot_path)
                
                # Check if content is allowed
                if not self.is_allowed(current_text):
                    not_allowed_count += 1
                    if not_allowed_count >= MAX_NOT_ALLOWED_TIME:
                        # Show warning
                        if not lock_warning_shown:
                            self.show_warning()
                            lock_warning_shown = True

                        if not final_lock_timer_start and self.warning_count >= MAX_NOTIFICATION_TIMES:
                            final_lock_timer_start = time.time()

                        if self.lock_timer_start is None:
                            self.lock_timer_start = time.time()
                        elif time.time() - self.lock_timer_start >= LOCK_INTERVAL_SECONDS:
                            self.lock_screen()

                            # Reset everything after locking
                            not_allowed_count = 0
                            lock_warning_shown = False
                            self.warning_count = 0
                            self.lock_timer_start = None
                            final_lock_timer_start = None
                else:
                    not_allowed_count = 0
                    lock_warning_shown = False
                    self.lock_timer_start = None
               
                # Delete the screenshot file
                os.remove(screenshot_path)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            self.cleanup_screenshots()

def main():
    # Check if tesseract is installed
    try:
        subprocess.run(['tesseract', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("Error: Tesseract is not installed. Please install Tesseract OCR first.")
        return

    monitor = ScreenMonitor()
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=monitor.cleanup_screenshots)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    # Start monitoring
    monitor.monitor()

if __name__ == "__main__":
    main()
