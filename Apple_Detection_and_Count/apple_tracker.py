from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Open the video file
video_path = "assests/apple_tree_5.mp4"
cap = cv2.VideoCapture(video_path)

# Define target resolution for resizing (width, height)
target_resolution = (1024, 720)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width, height = target_resolution

# Define the codec and create VideoWriter object
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Store the track history
track_history = defaultdict(lambda: [])
unique_track_ids = set()

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Resize the frame
        frame = cv2.resize(frame, target_resolution, interpolation=cv2.INTER_AREA)

        # Run YOLO11 tracking
        result = model.track(frame, persist=True, classes=[47], conf=0.5, iou=0.4, tracker="custom_botsort.yaml")[0]

        # Get boxes and track IDs
        if result.boxes and result.boxes.id is not None:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()
            unique_track_ids.update(track_ids)

            # Visualize result on the frame
            frame = result.plot()

            # Plot tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)

                # Draw tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=5)

            # Display unique object count with background
            count_text = f"Unique Objects: {len(unique_track_ids)}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            text_size = cv2.getTextSize(count_text, font, font_scale, thickness)[0]
            text_x, text_y = 10, 30
            # Draw background rectangle
            background_color = (0, 0, 0)  # Black background (BGR)
            rect_top_left = (text_x, text_y - text_size[1])
            rect_bottom_right = (text_x + text_size[0], text_y + 15)
            cv2.rectangle(frame, rect_top_left, rect_bottom_right, background_color, -1)
            # Draw text
            cv2.putText(frame, count_text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

        # Write the frame to the output video
        out.write(frame)

        # Display the frame
        cv2.imshow("YOLO11 Tracking", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Print final unique object count
print(f"Total Unique Objects Tracked: {len(unique_track_ids)}")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
