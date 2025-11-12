# Import required libraries
from ultralytics import YOLO
import cv2
import os

# 1️⃣ Load your trained model (.pt)
model_path = "turtlebot3_lane_detection.pt"
model = YOLO(model_path)

# 2️⃣ Set input and output video paths
input_video = "Raw_Datasets/video_2.mp4"   # path to your video
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)  # create folder if not exists
output_video = os.path.join(output_folder, "output_video2.mp4")

# 3️⃣ Open input video
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

frame_count = 0

# 4️⃣ Process video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model.predict(frame, conf=0.5, verbose=False)

    # Draw predictions on frame
    frame_with_boxes = results[0].plot()

    # Write frame to output video
    out.write(frame_with_boxes)

    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

# 5️⃣ Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Detection complete. Saved video to: {output_video}")
