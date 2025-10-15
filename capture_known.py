# capture_known.py
# usage: python capture_known.py <name>
# Example: python capture_known.py Alice

import sys
import cv2
import os
from helpers import ensure_dirs

def main():
    if len(sys.argv) < 2:
        print("Usage: python capture_known.py <name>")
        return
    name = sys.argv[1]
    ensure_dirs()
    cap = cv2.VideoCapture(0)
    count = 0
    print("Press 'c' to capture a photo for", name, ". Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Capture Known Face - press c", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            count += 1
            save_path = os.path.join("known_faces", f"{name}_{count}.jpg")
            cv2.imwrite(save_path, frame)
            print("Saved:", save_path)
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
