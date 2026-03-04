# ENGINEERING ASSESSMENT
**Role:** Computer Vision Engineer 
**Project:** Robust Multi-Object Tracking (MOT) in Dynamic Sports Environments

## 1. The Business Context
Clients require a high-fidelity analytics pipeline to track basketball players in real-time. The court is a highly chaotic environment:
* **Severe Occlusion:** Players constantly cross paths, set screens, and block the camera's line of sight.
* **Visual Similarity:** Both teams wear uniform colors, making appearance-based re-identification highly volatile.
* **Dynamic Camera:** The broadcast camera pans, tilts, and zooms. This is not a static CCTV feed.

We need a tracking system that maintains persistent unique identifiers (IDs) for every player on the court, regardless of these variables. 

## 2. Your Mission
Architect an end-to-end Object Detection and Tracking pipeline utilizing a YOLO-family model and ByteTrack. You are required to build the entire data preparation, training, inference, and evaluation pipeline from scratch.

## 3. Hard Engineering Constraints
I do not want a massive, monolithic Jupyter Notebook. I want a modular system. 

* **Data Pipeline:** Ingest the provided multi-sport MOT dataset. Programmatically isolate the basketball subset. Write a robust parser to convert `gt.txt` (MOT format) into normalized YOLO annotations. 
* **Detection Architecture:** Train your chosen YOLO model. You must justify your specific model choice (e.g., YOLOv8n vs YOLOv8m) based on a trade-off between mAP and inference speed.
* **Tracking Engine (ByteTrack):** Implement the ByteTrack algorithm. You are expected to manually tune the Kalman Filter parameters to handle the specific dynamics of human acceleration and deceleration on a basketball court.
* **Performance:** The inference pipeline must be capable of running at **>30 FPS** on a standard cloud GPU (e.g., T4 or V100). Latency matters.
* **Infrastructure:** The entire solution must be containerized using **Docker**. A single `docker-compose up` or `docker build/run` command should reproduce your environment perfectly.
* **Telemetry:** Integrate **Weights & Biases (W&B)**. Your submission must include a link to a public W&B dashboard detailing your training loss, validation mAP, and hyperparameter sweeps.

## 4. Edge Cases to Account For (The "Make or Break" Criteria)
Your tracking logic will be specifically tested against the following:
1.  **The "Screen" Test:** Two players cross paths, one completely obscuring the other for 5-10 frames. Does your tracker preserve the ID of the occluded player when they emerge?
2.  **Boundary Exits:** A player steps out of the camera frame and re-enters 3 seconds later. How does your system handle the ID assignment? (Document your assumptions here).

## 5. Required Deliverables
1.  **Codebase:** Clean, object-oriented, or strictly modular Python code.
2.  **`Dockerfile` & `requirements.txt`:** Flawless environment reproduction.
3.  **`README.md`:** Must include your architectural decisions, why you chose specific hyperparameters, and your W&B dashboard link.
4.  **Evaluation Metrics:** Calculate and output your final **MOTA** (Tracking Accuracy) and **IDF1** (ID F-Score) against the test set. 
5.  **Output Artifacts:** A `tracker_output.txt` (in MOT format) and an `.mp4` video showing the bounding boxes, confidence scores, and persistent IDs.