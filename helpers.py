# helpers.py
# Helper functions for face_detection.py

import cv2
import os
import numpy as np
import face_recognition
from datetime import datetime

# ------------------ Directories ------------------
KNOWN_FACES_DIR = "known_faces"
SNAPSHOT_DIR = "snapshots"

def ensure_dirs():
    """Ensure that necessary directories exist."""
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
    if not os.path.exists(SNAPSHOT_DIR):
        os.makedirs(SNAPSHOT_DIR)
    if not os.path.exists("emojis"):
        os.makedirs("emojis")
    if not os.path.exists("stickers"):
        os.makedirs("stickers")

# ------------------ Load Known Faces ------------------
def load_known_faces():
    """
    Load face encodings and names from KNOWN_FACES_DIR.
    Folder structure:
        known_faces/
            person1.jpg
            person2.jpg
    """
    encodings = []
    names = []

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith((".jpg", ".png")):
            img_path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(img_path)
            face_locations = face_recognition.face_locations(image)
            face_enc = face_recognition.face_encodings(image, face_locations)
            if face_enc:
                encodings.append(face_enc[0])
                names.append(os.path.splitext(filename)[0])
    return encodings, names

# ------------------ Save Snapshot ------------------
def save_snapshot(frame):
    """Save a snapshot of the current frame in SNAPSHOT_DIR with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SNAPSHOT_DIR, f"snapshot_{timestamp}.png")
    cv2.imwrite(path, frame)
    return path

# ------------------ Draw Label ------------------
def draw_label(frame, text, x, y, font_scale=0.5, color=(0,255,0), bg_color=(0,0,0)):
    """
    Draw text label with background rectangle on the frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, (x, y - h - 4), (x + w, y + 4), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
