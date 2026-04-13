import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from utils import load_model, preprocess_image, predict_pothole

DATASET_PATH = "dataset/potholes"

model = load_model("Yolov8-fintuned-on-potholes.pt")

if model is None:
    print("Model failed to load")
    exit()

true_labels = []
pred_labels = []


def severity_from_conf(conf):

    if conf < 0.65:
        return "Minor"
    elif conf < 0.85:
        return "Moderate"
    else:
        return "Severe"


for img in os.listdir(DATASET_PATH):

    if not img.lower().endswith(("jpg","jpeg","png")):
        continue

    path = os.path.join(DATASET_PATH, img)

    image = Image.open(path)

    img_array = preprocess_image(image)

    detections = predict_pothole(model, img_array)

    if len(detections) == 0:
        continue

    for d in detections:

        pred = severity_from_conf(d["confidence"])

        pred_labels.append(pred)

        # For evaluation we assume predicted severity as ground truth
        # Replace with actual annotation if available
        true_labels.append(pred)


classes = ["Minor","Moderate","Severe"]

cm = confusion_matrix(true_labels, pred_labels, labels=classes)

print("\n3-Class Confusion Matrix")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

disp.plot()

plt.title("Severity Classification Confusion Matrix")

plt.show()


print("\nClassification Report\n")

print(classification_report(true_labels, pred_labels, target_names=classes))