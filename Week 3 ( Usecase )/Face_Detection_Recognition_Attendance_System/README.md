[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://huggingface.co/spaces/24prathamesh2004/face-attendance-system)

# Face Detection & Recognition Attendance System

The system uses **YOLOv8-Face** for robust face detection and **ArcFace (buffalo_l)** for state-of-the-art face recognition, with attendance records stored securely in **MongoDB Atlas**.

---

## Features

### Multi-Pose Face Enrollment
- Captures **center, left, and right face profiles**
- Generates and stores high-quality face embeddings

### Real-Time Attendance
- Supports **webcam streaming** and **image-based input**
- Detects faces live and matches them against enrolled users
- Logs attendance with **timestamp and identity**

### Smart Attendance Logging
- Prevents duplicate attendance entries
- Uses a **configurable cooldown period** (e.g., 5 minutes)
- Ensures clean and reliable attendance records

### Attendance Reporting
- Filter attendance by **date and time**
- Export attendance records as **CSV**

---

##  Tech Stack

| Component | Technology |
|--------|-----------|
| Frontend | Streamlit |
| Face Detection | YOLOv8-Face |
| Face Recognition | InsightFace (ArcFace - buffalo_l) |
| Database | MongoDB Atlas |
| Backend | Python |
| Deployment | Hugging Face Spaces |
| Image Processing | OpenCV |

---






