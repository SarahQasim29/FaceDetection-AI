# FaceDetection-AI
Enhanced real-time face detection with emotion, age, gender, blink, smile, and gaze detection.

# Face Detection AI  

A Python-based **Face Detection System** built using **OpenCV** and **Deep Learning**, capable of identifying and marking human faces in real-time from webcam feed or image input.  

---

## 🚀 Features  

✅ **Real-Time Detection:** Detects faces instantly from webcam or video feed.  
✅ **Image Detection:** Identify faces in any uploaded image.  
✅ **Lightweight & Fast:** Optimized using OpenCV’s Haar Cascade and DNN models.  
✅ **Customizable:** Easily switch between detection models or integrate with attendance/surveillance systems.  

---

## Tech Stack  

- **Language:** Python  
- **Libraries:**  
  - `opencv-python`  
  - `numpy`  
  - `tensorflow` *(optional for advanced DNN models)*  
- **IDE:** VS Code / PyCharm  

---

## Installation  

### Clone the repository  
```bash
git clone https://github.com/SarahQasim29/FaceDetection-AI.git
cd FaceDetection-AI
-> Create and activate a virtual environment
bash
Copy code
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
-> Install dependencies
bash
Copy code
pip install -r requirements.txt
-> How to Run

-> For image-based face detection:
bash
Copy code
python face_detection_image.py
📹 For real-time webcam detection:
bash
Copy code
python face_detection.py
Detected faces will be highlighted with rectangles on the live feed or the processed image window.

**Project Structure**
bash
Copy code
FaceDetection-AI/
│
├── face_detection.py          # Main real-time detection script
├── face_detection_image.py    # Image-based detection
├── haarcascade_frontalface.xml # Haar model for face detection
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
└── examples/                  # Example input/output images (optional)
 Output Example
Input:
An image or webcam frame with one or more faces.

**Output:**
The same frame with green rectangles around detected faces.

 *Future Enhancements*
 Add mask detection

 Integrate emotion recognition

 Deploy using Streamlit or Flask web app

Add performance benchmarking

🧑‍💻 Author
Sarah Qasim
📍 GitHub Profile
💬 Open for collaborations and improvements!

🪪 License
This project is licensed under the MIT License – feel free to use and modify it with proper credit.
