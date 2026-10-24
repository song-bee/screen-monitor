"""
screen-watcher.py  (macOS only)

Every CAPTURE_INTERVAL_SECONDS it takes a screenshot, asks OpenAI if the
screen shows a video game or a novel, warns up to MAX_WARNINGS times,
then locks the screen.

Setup:
    pip install openai python-dotenv
    cp .env.example .env     # then put your OpenAI key in .env
    python screen-watcher.py

Settings live in .env (see .env.example).
"""

import base64
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import time
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Keep the OpenAI and HTTP libraries quiet. Only our own log() prints.
for name in ("openai", "httpx", "httpcore"):
    logging.getLogger(name).setLevel(logging.WARNING)

API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
INTERVAL = int(os.getenv("CAPTURE_INTERVAL_SECONDS", "30"))
MAX_WARNINGS = int(os.getenv("MAX_WARNINGS", "3"))
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "1280"))  # smaller = cheaper
IMAGE_DETAIL = os.getenv("IMAGE_DETAIL", "low")  # low, high or auto
CAPTURE_MODE = os.getenv("CAPTURE_MODE", "full").lower()  # full or crop
CROP_SIZE = min(1.0, max(0.1, float(os.getenv("CROP_SIZE", "0.5"))))
CROP_ZONE = min(1.0, max(CROP_SIZE, float(os.getenv("CROP_ZONE", "0.8"))))
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"

PROMPT = (
    "You are checking a screenshot of a person's computer screen. "
    "Answer whether the main thing on screen is a VIDEO GAME being played "
    "or a NOVEL / fiction story being read (web novel, e-book, light novel, "
    "manga, fan fiction). Ignore small ads or thumbnails. "
    'Reply with JSON only: {"bad": true or false, "reason": "short reason"}'
)

client = None  # created in main() after the key check
SCREEN = (0, 0)  # screen width and height, set in main()


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def screen_size():
    """Screen width and height in points, from Finder."""
    out = subprocess.run(
        ["osascript", "-e",
         'tell application "Finder" to get bounds of window of desktop'],
        capture_output=True, text=True, check=True,
    ).stdout
    left, top, right, bottom = [int(float(v)) for v in out.split(",")]
    return right - left, bottom - top


def random_crop_box(width, height):
    """Pick a crop box of CROP_SIZE of the screen, at a random spot that keeps
    it inside the middle CROP_ZONE of the screen. Every spot is equally likely."""
    cw, ch = int(width * CROP_SIZE), int(height * CROP_SIZE)
    zw, zh = int(width * CROP_ZONE), int(height * CROP_ZONE)
    zx, zy = (width - zw) // 2, (height - zh) // 2  # top-left of the zone
    x = random.randint(zx, zx + max(0, zw - cw))
    y = random.randint(zy, zy + max(0, zh - ch))
    return x, y, cw, ch


def capture_screen():
    """Take a screenshot (full or a random part) and shrink it."""
    path = os.path.join(tempfile.gettempdir(), "screen-watcher.jpg")
    cmd = ["screencapture", "-x", "-t", "jpg"]
    if CAPTURE_MODE == "crop" and CROP_SIZE < 1.0:
        x, y, w, h = random_crop_box(*SCREEN)
        cmd += ["-R", f"{x},{y},{w},{h}"]
    subprocess.run(cmd + [path], check=True)
    subprocess.run(["sips", "-Z", str(IMAGE_WIDTH), path],
                   check=True, capture_output=True)
    return path


def analyze(image_path):
    """Ask OpenAI. Returns (bad, reason)."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    resp = client.chat.completions.create(
        model=MODEL,
        store=False,  # ask OpenAI not to keep this request
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}",
                               "detail": IMAGE_DETAIL}},
            ],
        }],
    )
    data = json.loads(resp.choices[0].message.content)
    return bool(data.get("bad")), str(data.get("reason", ""))


def notify(title, message):
    def q(text):  # make the text safe inside AppleScript quotes
        return text.replace("\\", "\\\\").replace('"', '\\"')

    script = (f'display notification "{q(message)}" '
              f'with title "{q(title)}" sound name "Basso"')
    subprocess.run(["osascript", "-e", script])


def lock_screen():
    if DRY_RUN:
        log("DRY_RUN: would lock the screen now")
        return
    cgsession = ("/System/Library/CoreServices/Menu Extras/User.menu"
                 "/Contents/Resources/CGSession")
    if os.path.exists(cgsession):
        subprocess.run([cgsession, "-suspend"])
    else:
        subprocess.run(["pmset", "displaysleepnow"])
    log("Screen locked")


def main():
    if sys.platform != "darwin":
        sys.exit("This script only works on macOS.")
    if not API_KEY:
        sys.exit("OPENAI_API_KEY is missing. Put it in .env")
    global client, SCREEN
    client = OpenAI(api_key=API_KEY)
    SCREEN = screen_size()

    log(f"Watching every {INTERVAL}s, model={MODEL}, "
        f"max warnings={MAX_WARNINGS}, dry_run={DRY_RUN}")
    log(f"Screen {SCREEN[0]}x{SCREEN[1]}, capture={CAPTURE_MODE}, "
        f"crop_size={CROP_SIZE}, crop_zone={CROP_ZONE}, detail={IMAGE_DETAIL}")
    strikes = 0

    while True:
        time.sleep(INTERVAL)
        try:
            path = capture_screen()
            bad, reason = analyze(path)
            os.remove(path)
        except Exception as e:
            log(f"Error: {e}")
            continue

        if not bad:
            if strikes:
                log("Screen is fine again, warnings reset")
            strikes = 0
            continue

        strikes += 1
        if strikes <= MAX_WARNINGS:
            log(f"Warning {strikes}/{MAX_WARNINGS}: {reason}")
            left = MAX_WARNINGS - strikes
            tail = (f"{left} more and the screen locks" if left
                    else "next time the screen locks")
            notify(f"Screen warning {strikes}/{MAX_WARNINGS}",
                   f"{reason}. {tail}.")
        else:
            log(f"Locking: {reason}")
            notify("Locking screen", reason)
            lock_screen()
            strikes = 0


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Stopped")
