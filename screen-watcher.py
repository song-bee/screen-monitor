"""
screen-watcher.py  (macOS only)

Every CAPTURE_INTERVAL_SECONDS it takes a screenshot, asks OpenAI if the
screen shows one of the BLOCKED_CATEGORIES (games, novels, video, ...),
warns up to MAX_WARNINGS times, then locks the screen.

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
NOTIFY_SECONDS = int(os.getenv("NOTIFY_SECONDS", "15"))  # 0 = normal banner
BLOCKED_CATEGORIES = os.getenv(
    "BLOCKED_CATEGORIES",
    "video game; novel (web novel, e-book, light novel, manga, fan fiction)",
)


def parse_categories(text):
    """Split 'name (examples); name (examples)' into a list of
    (name, full text). The name is the part before any '('."""
    cats = []
    for item in text.split(";"):
        item = item.strip()
        name = item.split("(")[0].strip().lower()
        if name and name != "none" and name not in [c[0] for c in cats]:
            cats.append((name, item))
    return cats


CATEGORIES = parse_categories(BLOCKED_CATEGORIES)
CATEGORY_NAMES = [name for name, _ in CATEGORIES]

PROMPT = (
    "You are checking a screenshot of a person's computer screen. "
    "First, in 'reason', say in a few words what the main thing on screen is. "
    "Then set 'category' to the blocked category below that matches that "
    "main thing, or 'none' if it matches none of them. "
    "Blocked categories: " + "; ".join(item for _, item in CATEGORIES) + ". "
    "Anything not in that list is 'none'. Ignore small ads or thumbnails."
)

# Strict schema: the model must return exactly these two fields, the
# reason first, and category must be one of our names or "none".
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "screen_check",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "category": {"type": "string",
                             "enum": CATEGORY_NAMES + ["none"]},
            },
            "required": ["reason", "category"],
            "additionalProperties": False,
        },
    },
}

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
    """Ask OpenAI. Returns (bad, reason, raw answer text)."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    resp = client.chat.completions.create(
        model=MODEL,
        store=False,  # ask OpenAI not to keep this request
        response_format=RESPONSE_FORMAT,
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
    raw = resp.choices[0].message.content or ""
    return parse_answer(raw) + (raw,)


def parse_answer(raw):
    """Read the model's JSON. Bad only if category is one of ours."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"answer is not JSON: {raw!r}")
    if not isinstance(data, dict):
        raise ValueError(f"answer is not a JSON object: {raw!r}")
    category = data.get("category")
    reason = str(data.get("reason", "")).strip()
    if isinstance(category, str) and category.strip().lower() in CATEGORY_NAMES:
        return True, f"{category.strip().lower()}: {reason}"
    return False, reason


def notify(title, message):
    def q(text):  # make the text safe inside AppleScript quotes
        return text.replace("\\", "\\\\").replace('"', '\\"')

    if NOTIFY_SECONDS <= 0:
        # Normal banner. macOS closes it after about 5 seconds.
        script = (f'display notification "{q(message)}" '
                  f'with title "{q(title)}" sound name "Basso"')
        subprocess.run(["osascript", "-e", script])
        return

    # A dialog box that stays up for NOTIFY_SECONDS, then closes itself.
    # Popen so the watcher keeps running while the box is open.
    script = (f'display dialog "{q(message)}" with title "{q(title)}" '
              f'buttons {{"OK"}} default button "OK" with icon caution '
              f'giving up after {NOTIFY_SECONDS}')
    subprocess.Popen(["osascript", "-e", "beep", "-e", script],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
    if not CATEGORIES:
        sys.exit("BLOCKED_CATEGORIES is empty. Put at least one in .env")
    global client, SCREEN
    client = OpenAI(api_key=API_KEY, timeout=60, max_retries=1)
    SCREEN = screen_size()

    log(f"Watching every {INTERVAL}s, model={MODEL}, "
        f"max warnings={MAX_WARNINGS}, dry_run={DRY_RUN}")
    log(f"Screen {SCREEN[0]}x{SCREEN[1]}, capture={CAPTURE_MODE}, "
        f"crop_size={CROP_SIZE}, crop_zone={CROP_ZONE}, detail={IMAGE_DETAIL}")
    log(f"Blocked: {', '.join(CATEGORY_NAMES)}")
    strikes = 0

    first = True
    while True:
        if not first:
            time.sleep(INTERVAL)
        first = False

        start = time.time()
        try:
            path = capture_screen()
            bad, reason, raw = analyze(path)
            os.remove(path)
        except Exception as e:
            log(f"Error after {time.time() - start:.1f}s: {e}")
            continue
        took = time.time() - start
        raw = raw.replace("\n", " ")  # keep each log entry on one line

        if not bad:
            log(f"OK ({took:.1f}s) raw={raw}")
            if strikes:
                log("Screen is fine again, warnings reset")
            strikes = 0
            continue

        strikes += 1
        if strikes <= MAX_WARNINGS:
            log(f"Warning {strikes}/{MAX_WARNINGS} ({took:.1f}s) raw={raw}")
            left = MAX_WARNINGS - strikes
            tail = (f"{left} more and the screen locks" if left
                    else "next time the screen locks")
            notify(f"Screen warning {strikes}/{MAX_WARNINGS}",
                   f"{reason}. {tail}.")
        else:
            log(f"Locking ({took:.1f}s) raw={raw}")
            notify("Locking screen", reason)
            lock_screen()
            strikes = 0


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Stopped")
