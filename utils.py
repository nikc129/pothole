import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os


def load_model(model_path="Yolov8-fintuned-on-potholes.pt"):
    """
    Load YOLOv8 pothole detection model from local file
    """

    if not os.path.exists(model_path):
        print("Model file not found:", model_path)
        return None

    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None


def preprocess_image(image):
    """
    Convert PIL Image to numpy format
    """

    img_array = np.array(image)

    if img_array.shape[-1] == 4:  # RGBA → RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    elif len(img_array.shape) == 2:  # Grayscale → RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

    return img_array


def predict_pothole(model, image_array, conf_threshold=0.5):
    """
    Run YOLO pothole detection
    """

    results = model(image_array, conf=conf_threshold)

    detections = []

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:

                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()

                detections.append({
                    "confidence": confidence,
                    "bbox": bbox,
                    "severity": get_severity(confidence)
                })

    return detections


def get_severity(confidence):

    if confidence < 0.65:
        return "Low"
    elif confidence < 0.85:
        return "Medium"
    else:
        return "Severe"


def draw_detections(image_array, detections):

    for det in detections:

        x1, y1, x2, y2 = map(int, det["bbox"])
        conf = det["confidence"]
        severity = det["severity"]

        label = f"Pothole {conf:.2f} ({severity})"

        cv2.rectangle(image_array, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.putText(
            image_array,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    return image_array