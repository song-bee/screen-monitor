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
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"

PROMPT = (
    "You are checking a screenshot of a person's computer screen. "
    "Answer whether the main thing on screen is a VIDEO GAME being played "
    "or a NOVEL / fiction story being read (web novel, e-book, light novel, "
    "manga, fan fiction). Ignore small ads or thumbnails. "
    'Reply with JSON only: {"bad": true or false, "reason": "short reason"}'
)

client = None  # created in main() after the key check


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def capture_screen():
    """Take a screenshot with the built-in macOS tool and shrink it."""
    path = os.path.join(tempfile.gettempdir(), "screen-watcher.jpg")
    subprocess.run(["screencapture", "-x", "-t", "jpg", path], check=True)
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
                               "detail": "low"}},
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
    global client
    client = OpenAI(api_key=API_KEY)

    log(f"Watching every {INTERVAL}s, model={MODEL}, "
        f"max warnings={MAX_WARNINGS}, dry_run={DRY_RUN}")
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
