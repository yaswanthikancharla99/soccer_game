# 🏆 Player Re-Identification using YOLOv11 + SORT

An end-to-end computer vision pipeline for real-time **player detection**, **tracking**, and **re-identification** using a single video feed. This project combines **YOLOv11** with **SORT** to ensure consistent ID assignment and tracking accuracy throughout a 15-second sports clip.

---

## 🎯 Project Objective

Simulate intelligent player tracking by:

- 🔍 Detecting players using a fine-tuned YOLOv11 model  
- 🆔 Assigning unique IDs based on early video frames  
- 🔁 Maintaining identities even when players leave and re-enter the frame  
- 📊 Exporting frame-wise tracking data for analysis  

---

## 🚀 Technologies Used

| Tool / Library           | Purpose                                   |
|--------------------------|-------------------------------------------|
| `Ultralytics YOLOv11`    | Fast, accurate object detection           |
| `SORT`                   | Realtime tracking via Kalman Filter       |
| `OpenCV`                 | Video processing and frame annotation     |
| `NumPy`                  | Numerical computation                     |
| `FilterPy`               | Filtering module for SORT tracker         |
| `scikit-image`           | Image utility operations                  |
| `Python`                 | Core implementation language              |

---

## 📁 Project Structure

📂 soccer_game/ ├── yash.py # Main tracking script ├── players_yolov11.pt # Fine-tuned YOLOv11 model ├── 15sec_input_720p.mp4 # Input sports video clip ├── output_tracking.mp4 # Output with annotated player IDs ├── tracking_data.csv # CSV log with player positions └── sort/ └── sort.py # SORT tracking implementation



---

## 🔧 Installation Guide

### Clone the Repository

```bash
git clone https://github.com/yaswanthikancharla99/soccer_game.git
cd soccer_game

pip install ultralytics opencv-python numpy filterpy scikit-image


python yash.py



Frame	Player ID	x1	y1	x2	y2	Confidence
1	0	104	73	165	142	0.88
1	1	220	76	283	144	0.92



💡 Features
⚡ Real-time detection and re-identification

↩️ Robust tracking across occlusions and re-entries

📊 Exportable data for heatmaps, trajectories, and match analysis

🧠 Ready for enhancement with DeepSORT or appearance features
