from ultralytics import YOLO
from src.utils.path_utils import load_config
import cv2
import numpy as np


class YOLOv12Detector:
    def __init__(self):
        """Initialize YOLOv12 detector with configuration parameters"""
        self.model_config = load_config("model")
        self.dataset_config = load_config("dataset")

        # Load pre-trained YOLOv12 model
        self.model = YOLO(self.model_config["model_path"])
        self.conf_thres = self.model_config["conf_threshold"]
        self.iou_thres = self.model_config["iou_threshold"]
        self.class_names = self.dataset_config["names"]

    def detect(self, image_path: str) -> tuple[np.ndarray, list]:
        """
        Perform object detection on input image
        :param image_path: Path to input image
        :return: (original image array, detection results list)
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image from: {image_path}")

        # Model inference
        results = self.model(
            img,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=self.model_config["imgsz"]
        )

        # Parse detection results
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                # Extract class info and confidence
                cls_id = int(box.cls[0])
                cls_name = self.class_names[cls_id]
                conf = float(box.conf[0])

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf
                })

        return img, detections

    def draw_detections(self, img: np.ndarray, detections: list) -> np.ndarray:
        """
        Draw bounding boxes and labels on image
        :param img: Original image array
        :param detections: Detection results list
        :return: Image with drawn detections
        """
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["class_name"]
            conf = det["confidence"]

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label
            cv2.putText(
                img,
                f"{cls_name} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        return img