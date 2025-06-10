# Accident-Detection-and-Response-System-ResNet-Final

A real-time accident detection and response system using ResNet deep learning algorithm. This final year project uses a live camera feed to detect and classify road accidents as major, minor, or none, and then triggers emergency alerts.

---

## 📌 Features

- Real-time video stream using OpenCV
- Accident severity classification (Major, Minor, None)
- ResNet deep learning model
- SSD-MobileNet for human detection
- Flask web interface
- Email alert functionality
- SQLite-based accident logging

---

## 🛠 Technologies Used

- Python
- Flask
- TensorFlow / Keras
- OpenCV
- NumPy, Pillow
- HTML/CSS
- SQLite

---

## 🗂 Project Folder Structure
accident-detection/
├── app.py
├── detection.py
├── model_weights.keras <-- (not included in repo)
├── model.json <-- (ResNet model architecture)
├── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
├── templates/
├── static/
├── .gitignore
├── requirements.txt
└── README.md

