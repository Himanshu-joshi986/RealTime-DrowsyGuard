# 🚗 RealTime‑DrowsyGuard – Hybrid Driver Drowsiness Detection System

> **Real‑time hybrid driver drowsiness detection system** using **EAR, PERCLOS, MAR, Head Pose** with auto‑calibration and a weighted fatigue score (0–100).

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-yellow.svg)](https://opencv.org/)

**About**  
Real‑Time Hybrid Driver Drowsiness Detection System is a lightweight, webcam‑based solution that monitors driver fatigue using classical computer vision (EAR, PERCLOS, MAR, and head pose) with Google Mediapipe facial landmarks. The system auto‑calibrates to your personal eye and mouth baselines, fuses four fatigue signals into a 0–100 weighted score, and triggers an audio alarm only when sustained drowsiness is detected—making it ideal for low‑cost, real‑time driver monitoring.  
Main repo: [RealTime‑DrowsyGuard](https://github.com/Himanshu-joshi986/RealTime-DrowsyGuard)

---

## ✨ **Unique Features**

- **Auto‑calibration** – Learns your personal eye & mouth baselines in **4 seconds**, eliminating false alarms.
- **Distance‑adaptive head pose** – Thresholds automatically increase when the face is far from the camera.
- **Deadzone filtering** – Ignores small head movements (talking, glancing) to prevent false triggers.
- **3‑pair MAR** – Stable yawn detection that doesn’t fire on slight mouth openings or face tilt.
- **Weighted fatigue score (0–100)** – Fuses four signals with hysteresis‑controlled alarm.
- **Live HUD** – Real‑time metrics, graphs, event log, and session stats.
- **No deep learning required** – Runs fast on any laptop with a webcam.

---

## 🧠 **Technical Stack**

| Component                 | Technology |
|---------------------------|------------|
| Face & landmark detection | [Google Mediapipe Face Landmarker](https://developers.google.com/mediapipe) (468 points) |
| Eye closure               | EAR (Eye Aspect Ratio) + PERCLOS rolling window |
| Yawn detection            | 3‑pair MAR (Mouth Aspect Ratio) |
| Head pose                 | OpenCV `solvePnP` with a 3D face model |
| Smoothing                 | Exponential moving average (EMA) on all signals |
| Alarm                     | Pygame mixer + `alarm.wav` |
| GUI                       | OpenCV overlays |

---

## 📁 **Project Structure**
DROWSINESS-ALERT-SYSTEM/
├── main.py # Main loop, HUD, alarm logic
├── ear_mar_utils.py # EAR, MAR, RollingMetric
├── head_pose.py # Head pose estimator
├── alarm.wav # Alert sound (place in root)
├── requirements.txt # Python dependencies
├── README.md # This file
└── models/
└── face_landmarker.task # Auto‑downloaded on first run (~3 MB)

text

---

## 🚀 **Quick Start**

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Run the System**
```bash
python main.py
```

### 3. **Calibration**
- Keep your **face centered**, **eyes open**, **mouth closed** during the **4‑second calibration**.
- The face landmarker model (~3 MB) downloads automatically on first run.
- Press **Q** to quit.

---

## 🎛️ **Configurable Thresholds**

Edit values in `main.py`:

| Parameter                   | Default | Description |
|-----------------------------|---------|-------------|
| `CALIB_SECONDS`             | `4`     | Calibration duration |
| `EAR_FRAMES`                | `10`    | Frames eyes must stay closed (~0.33 sec) |
| `HEAD_PITCH_THRESH`         | `14°`   | Forward nod threshold (adaptive up to 28° when far) |
| `HEAD_ROLL_THRESH`          | `18°`   | Side tilt threshold |
| `SCORE_ALARM`               | `55`    | Fatigue score to trigger alarm |
| `W_EAR / W_PERCLOS / W_YAWN / W_HEAD` | `40/30/10/20` | Weight distribution |

---

## 📊 **Fatigue Score Algorithm**

**Weighted combination (0–100):**  
\[
\text{Total Score} = 
(\text{EAR} \times 40\%) + 
(\text{PERCLOS} \times 30\%) + 
(\text{MAR} \times 10\%) + 
(\text{Head Pose} \times 20\%)
\]

- **EAR (40%)** – Instantaneous eye openness vs. your calibrated baseline.
- **PERCLOS (30%)** – % of closed‑eye frames in the last 2 seconds.
- **MAR (10%)** – Mouth opening ratio (3‑pair average).
- **Head Pose (20%)** – Forward/side tilt magnitude (distance‑adjusted).

**Alarm Logic:**  
- Activates when **score ≥ 55**.
- Stays on until **score drops below 25** (hysteresis).

---

## 🎓 **Academic Summary**

> "This system combines **classical computer vision** (EAR, MAR, solvePnP) with **Mediapipe’s facial landmarks** to detect drowsiness through four independent signals. **Auto‑calibration** personalizes thresholds, **distance adaptation** ensures reliable operation at any camera distance, and a **weighted fatigue score with hysteresis** prevents false alarms. No deep learning or Haar cascades are used, making it lightweight and real‑time."

---

## 📋 **Requirements**

- **Python 3.8+**
- **Webcam**
- **Windows / Linux / macOS**

### Python packages

Install exactly what your project uses:

```text
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
scipy>=1.11.0
pygame>=2.5.0
```

You can install them with:

```bash
pip install -r requirements.txt
```

---

## 🔊 **Audio Setup**

1. Place `alarm.wav` in the root directory.
2. Or replace it with your preferred alert sound in `main.py`.

---

## 📄 **License**

[MIT License](LICENSE) – Free to use, modify, and distribute.

---

## 🙌 **Contributing**

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📞 **Support**

- Found a bug? [Open an issue](https://github.com/Himanshu-joshi986/RealTime-DrowsyGuard/issues)
- Need help? Check the [threshold tuning guide](#-configurable-thresholds).

---

**Built with ❤️ for road safety**