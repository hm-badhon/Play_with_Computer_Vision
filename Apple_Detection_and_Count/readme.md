
# 🍎 Apple Tracking in Orchard Videos using YOLOv11 + BoT-SORT

This project implements a real-time object detection and tracking pipeline using the **YOLOv11** model and **BoT-SORT** tracker to monitor apples in orchard footage. It identifies apples, tracks them across frames with unique IDs, and overlays their paths and count in an annotated output video.


## 🚀 Features

- ✅ Real-time object detection with **YOLOv11n**
- 🎯 Accurate tracking using **BoT-SORT**
- 🔄 Tracks apples with consistent IDs across frames
- 🧮 Displays real-time **unique object count**
- 🎥 Outputs annotated video with bounding boxes, trails, and overlay text

---




---

## 🧠 Model Details

- **Model**: YOLOv11n
- **Tracked Class**: Apple (class ID 47)
- **Tracker**: BoT-SORT (custom configuration)
- **Input Size**: Resized to `(1024, 720)`
- **Confidence Threshold**: 0.5
- **IoU Threshold**: 0.4

---

## 🔧 Requirements

Install dependencies with:

```bash
pip install ultralytics opencv-python numpy
````

---

## ▶️ How to Run

1. Clone the repository and navigate to the project folder.
2. Place your input video in the `test_video/` folder.
3. Run the script:

```bash
python apple_tracker.py
```

4. Press `q` to exit early. The final output will be saved as `output_video.mp4`.

---

## 📊 Output Example

* ✅ Bounding boxes for detected apples
* 🎨 Tracking trails for each object ID
* 📈 Live count of unique apples displayed on the video

---

## 🧩 Customization

* Change `classes=[47]` in `model.track(...)` to track different object classes.
* Adjust frame resolution via `target_resolution`.
* Modify `custom_botsort.yaml` for different tracking behavior.

---

## 📌 Applications

* Precision Agriculture
* Yield Estimation
* Robotic Fruit Harvesting
* Smart Orchard Monitoring

---

## 📬 Contact

Feel free to reach out if you're interested in extending this system for other crops, integrating drone feeds, or deploying it on edge devices!

---


## ⭐ Acknowledgements

* [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
* [BoT-SORT](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)

