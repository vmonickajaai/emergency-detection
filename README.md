🚨 Multimodal AI-Based Emergency Detection System

📖 Description

This project is a real-time emergency detection system that uses computer vision and audio analysis to identify critical situations such as fire, accidents, weapons, theft, and human falls.


The system combines YOLOv8 object detection with voice-based keyword recognition to improve accuracy and reduce false alarms. Once an emergency is detected, it triggers alerts and sends relevant information such as image and location.


🎯 Features
Real-time video monitoring (CCTV/Webcam)
Emergency detection using YOLOv8
Supports multiple classes:
Fire
Accident
Weapon
Theft
Falling person
Voice-based emergency detection (keywords like help, fire)
Alarm system for instant alerts
Sends captured image and location
Web interface using Flask


🛠️ Tech Stack
Language: Python
Framework: Flask
Model: YOLOv8 (Ultralytics)
Libraries: OpenCV, NumPy, Pandas
Frontend: HTML, CSS, Bootstrap
Audio Processing: SpeechRecognition


🏗️ Project Structure
emergency-detection/
│
├── models/              # Trained YOLOv8 weights
├── dataset/             # Training data
├── static/              # CSS, JS files
├── templates/           # HTML files
│
├── app.py               # Main Flask app
├── detect.py            # Detection logic
├── audio.py             # Voice detection
├── utils.py             # Helper functions
│
├── requirements.txt
└── README.md


⚙️ Installation
1. Clone Repository
git clone https://github.com/your-username/emergency-detection.git
cd emergency-detection
2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows
3. Install Dependencies
pip install -r requirements.txt
4. Run Application
python app.py


🤖 Model Training
Why YOLOv8?
Fast and efficient for real-time detection
High accuracy
Easy custom training
Training Command
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50


⚡ How It Works
Captures video frames from camera
Detects objects using YOLOv8
Listens for emergency keywords
Combines video + audio results
If emergency detected:
Triggers alarm
Captures image
Sends alert with location


📊 Results
Achieved ~90%+ detection accuracy
Real-time performance
Reduced false positives using multimodal approach


🚀 Future Improvements
Mobile application integration
Cloud deployment (AWS/GCP)
Live GPS tracking
Face recognition
Integration with emergency services


👩‍💻 Author

V MONICKA JAAI
B.Tech – AI & Data Science
