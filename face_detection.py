# face_detection.py
# Enhanced student-style face detection with emotion, age, gender, blink, smile, gaze detection
# Run: python face_detection.py

import cv2
import mediapipe as mp
import time
from deepface import DeepFace
import face_recognition
import numpy as np
from helpers import load_known_faces, save_snapshot, draw_label, ensure_dirs

# ------------------ initialize stuff ------------------
ensure_dirs()
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

# Initialize face mesh for AR / blink / gaze
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True,       # enables iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------------ Load known faces ------------------
known_encodings, known_names = load_known_faces()
print("Known faces loaded:", known_names)

# ------------------ Config flags ------------------
USE_DEEPFACE = True         # enable emotion/age/gender analysis
USE_RECOGNITION = True      # enable known face recognition
SAVE_ON_DETECT = False      # save snapshot when face is detected

# ------------------ Helper functions ------------------

def blink_ratio(landmarks, w, h):
    """
    Returns a simple blink ratio using eye landmarks.
    Left eye: [33, 133, 159, 145] | Right eye: [362, 263, 386, 374]
    """
    def eye_ratio(eye):
        # horizontal distance
        hor = np.linalg.norm(np.array([eye[0].x* w, eye[0].y* h]) -
                             np.array([eye[1].x* w, eye[1].y* h]))
        # vertical distance
        ver = np.linalg.norm(np.array([eye[2].x* w, eye[2].y* h]) -
                             np.array([eye[3].x* w, eye[3].y* h]))
        return ver / hor

    left = eye_ratio([landmarks[33], landmarks[133], landmarks[159], landmarks[145]])
    right = eye_ratio([landmarks[362], landmarks[263], landmarks[386], landmarks[374]])
    return (left + right) / 2

def mouth_aspect_ratio(landmarks, w, h):
    """
    Simple smile detection using mouth landmarks.
    Upper lip: 13, lower lip: 14, corners: 78, 308
    """
    hor = np.linalg.norm(np.array([landmarks[78].x* w, landmarks[78].y* h]) -
                         np.array([landmarks[308].x* w, landmarks[308].y* h]))
    ver = np.linalg.norm(np.array([landmarks[13].x* w, landmarks[13].y* h]) -
                         np.array([landmarks[14].x* w, landmarks[14].y* h]))
    return ver / hor

def gaze_direction(landmarks, w, h):
    """
    Approximate gaze: left, right, center
    Uses iris and eye center
    """
    if len(landmarks) > 473:  # ensure iris landmarks exist
        left_iris = landmarks[468]
        right_iris = landmarks[473]
        eye_center_x = (landmarks[33].x + landmarks[263].x) / 2
        iris_x = (left_iris.x + right_iris.x) / 2
        if iris_x < eye_center_x - 0.01:
            return "Looking Left"
        elif iris_x > eye_center_x + 0.01:
            return "Looking Right"
    return "Looking Center"

# ------------------ Start webcam ------------------
cap = cv2.VideoCapture(0)
prev_time = 0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Cannot read camera")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        annotated = frame.copy()

        # ------------------ Mediapipe Holistic ------------------
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(img_rgb)

        # Draw pose / hands landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(annotated, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(annotated, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # ------------------ Face recognition ------------------
        rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
        info_texts = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            cv2.rectangle(annotated, (left, top), (right, bottom), (0, 255, 0), 2)

            # Recognition
            name = "Unknown"
            if USE_RECOGNITION and known_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_idx = np.argmin(face_distances) if len(face_distances) > 0 else None
                if best_idx is not None and matches[best_idx]:
                    name = known_names[best_idx]

            # ------------------ Emotion, age, gender ------------------
            emotion_text = ""
            if USE_DEEPFACE:
                try:
                    pad = 30
                    top_c = max(0, top - pad)
                    left_c = max(0, left - pad)
                    bottom_c = min(h, bottom + pad)
                    right_c = min(w, right + pad)
                    face_img = frame[top_c:bottom_c, left_c:right_c]
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    face_img = cv2.resize(face_img, (224, 224))

                    analysis = DeepFace.analyze(
                        face_img,
                        actions=['emotion', 'age', 'gender'],
                        detector_backend='retinaface',
                        enforce_detection=False
                    )

                    result = analysis[0] if isinstance(analysis, list) else analysis
                    dominant_emotion = result.get('dominant_emotion', '')
                    gender = result.get('dominant_gender', '')
                    age = result.get('age', '')
                    emotion_text = f"{dominant_emotion}, {gender}, {age}"

                except Exception:
                    emotion_text = "Emotion not detected"

            draw_label(annotated, name, left, top)
            if emotion_text:
                draw_label(annotated, emotion_text, left, bottom + 20, bg_color=(50, 50, 50))
            info_texts.append(name + (" | " + emotion_text if emotion_text else ""))

            if SAVE_ON_DETECT:
                save_path = save_snapshot(annotated)
                print("Saved snapshot as", save_path)

        # ------------------ Face Mesh for blink, gaze, smile ------------------
        mesh_results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                # Blink
                blink = blink_ratio(face_landmarks.landmark, w, h)
                blink_text = "Eyes Closed" if blink < 0.25 else "Eyes Open"

                # Smile
                mar = mouth_aspect_ratio(face_landmarks.landmark, w, h)
                smile_text = "Smiling" if mar > 0.35 else "Neutral"

                # Gaze
                gaze_text = gaze_direction(face_landmarks.landmark, w, h)

                # Show on frame
                draw_label(annotated, blink_text, 10, h - 80)
                draw_label(annotated, smile_text, 10, h - 60)
                draw_label(annotated, gaze_text, 10, h - 40)

        # ------------------ FPS & Info HUD ------------------
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0.0
        prev_time = curr_time

        cv2.putText(annotated, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f"Faces: {len(face_locations)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        y = 90
        for line in info_texts[:5]:
            cv2.putText(annotated, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y += 25

        cv2.imshow("Student Face Detection AI", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            p = save_snapshot(annotated)
            print("Snapshot saved:", p)

cap.release()
cv2.destroyAllWindows()
