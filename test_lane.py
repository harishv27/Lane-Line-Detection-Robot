# ==========================================
# 🧠 Simplified Lane Direction Control (YOLO + OpenCV)
# Turns if car is >5cm or >10cm from lane center (approx pixels)
# ==========================================
from ultralytics import YOLO
import cv2
import numpy as np
import os

# 1️⃣ Load YOLO lane detection model
model_path = "turtlebot3_lane_detection.pt"
model = YOLO(model_path)

# 2️⃣ Input & output
input_video = "Raw_Datasets/video_2.mp4"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)
output_video = os.path.join(output_folder, "lane_direction_simple.mp4")

# 3️⃣ Video capture & writer
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# 4️⃣ Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO lane detection
    results = model.predict(frame, conf=0.5, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []

    frame_with_boxes = results[0].plot()
    lane_centers = []

    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        cx = int((x1 + x2) / 2)
        lane_centers.append(cx)
        cv2.circle(frame_with_boxes, (cx, int((y1+y2)/2)), 5, (0, 255, 0), -1)

    # Default direction
    direction = "No Lane Detected"

    if len(lane_centers) >= 2:
        avg_lane_center = int(np.mean(lane_centers))
        frame_center = width // 2

        # Simple distance-based threshold (approx pixels)
        distance_pixels = avg_lane_center - frame_center

        # Thresholds: ~5cm = 50px, ~10cm = 100px (adjust based on camera)
        if abs(distance_pixels) < 50:
            direction = "FORWARD"
            color = (0, 255, 0)
        elif abs(distance_pixels) < 100:
            direction = "SLIGHT TURN LEFT" if distance_pixels < 0 else "SLIGHT TURN RIGHT"
            color = (0, 255, 255)
        else:
            direction = "TURN LEFT" if distance_pixels < 0 else "TURN RIGHT"
            color = (0, 0, 255)

        # Draw guide line
        cv2.line(frame_with_boxes, (frame_center, height), (avg_lane_center, height//2), color, 3)

    # Overlay direction
    cv2.putText(frame_with_boxes, f"Direction: {direction}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # Save frame
    out.write(frame_with_boxes)

# 5️⃣ Cleanup
cap.release()
out.release()
print(f"✅ Output saved to: {output_video}")
