# Basketball Player Tracking

A modular, Dockerized multi-object tracking pipeline for basketball broadcast video. A YOLO11m detector feeds ByteTrack (via boxmot) with a Kalman filter tuned for fast player motion (position noise 1/15, velocity noise 1/100), trained and evaluated on the SportsMOT basketball subset.

- [Architecture](architecture.html)
- [Source and full README](https://github.com/The-Harsh-Vardhan/Basketball-Player-Tracking)

Status: the pipeline, Docker setup and unit tests are in place. A recorded training run, MOTA/IDF1 results and a public W&B dashboard are not yet published (tracked in the repository issues).
