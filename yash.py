from ultralytics import YOLO
from sort.sort import Sort
import cv2
import numpy as np
import csv

# Load YOLOv11 model
model = YOLO('players_yolov11.pt')

# Initialize SORT tracker
tracker = Sort()

# Load video
cap = cv2.VideoCapture('15sec_input_720p.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video writer
out = cv2.VideoWriter('output_tracking.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1

    # YOLOv11 inference
    results = model(frame)[0]
    boxes = results.boxes

    detections = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        detections.append([x1, y1, x2, y2, conf])

    detections = np.array(detections)

    # Handle frames with no detections
    if detections.shape[0] == 0:
        detections = np.empty((0, 5))

    # Update tracker
    track_results = tracker.update(detections)

    for track in track_results:
        x1, y1, x2, y2, track_id = track.astype(int)

        # Draw bounding boxes and IDs
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Player {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Optional: Crop for embedding (appearance matching)
        # cropped = frame[y1:y2, x1:x2]
        # embedding = resnet_model(cropped)  ← Placeholder for future use

        # Save detection data
        with open('tracking_data.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([frame_num, track_id, x1, y1, x2, y2, conf])

    out.write(frame)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Finished! Output saved as 'output_tracking.mp4' and 'tracking_data.csv'")
