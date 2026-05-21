from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# COCO class IDs for vehicles
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}


class VehicleTracker:
    def __init__(self, model_size='n', conf=0.4, target_classes=None):
        print(f"Loading YOLOv8{model_size}... (downloads ~6MB on first run)")
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.tracker = DeepSort(
            max_age=30,       # frames before a lost track is dropped
            n_init=2,         # detections needed before track is confirmed
            nms_max_overlap=1.0,
            max_cosine_distance=0.4,
        )
        self.conf = conf
        self.target_classes = target_classes or set(VEHICLE_CLASSES.keys())

    def update(self, frame):
        """Run detection + tracking. Returns list of DeepSort Track objects."""
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls not in self.target_classes or conf < self.conf:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # DeepSort expects [left, top, width, height]
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        return self.tracker.update_tracks(detections, frame=frame)
