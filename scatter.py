import os
import shutil
from PIL import Image
from utils import load_model, preprocess_image, predict_pothole

# Paths
NORMAL_DIR = "dataset/normal"
POTHOLE_DIR = "dataset/potholes"

OUTPUT_TRUE = "true"
OUTPUT_FALSE = "false"

MODEL_PATH = "Yolov8-fintuned-on-potholes.pt"

TARGET_COUNT = 50


def create_dirs():
    os.makedirs(OUTPUT_TRUE, exist_ok=True)
    os.makedirs(OUTPUT_FALSE, exist_ok=True)


def is_pothole_detected(detections):
    """
    If at least one detection exists → pothole = True
    """
    return len(detections) > 0


def process_folder(folder_path, expected_label, output_path, model):
    """
    expected_label:
        True → pothole expected
        False → no pothole expected
    """

    count = 0

    for img_name in os.listdir(folder_path):
        if count >= TARGET_COUNT:
            break

        img_path = os.path.join(folder_path, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            continue

        img_array = preprocess_image(image)
        detections = predict_pothole(model, img_array)

        prediction = is_pothole_detected(detections)

        # Match expected condition
        if prediction == expected_label:
            dst_path = os.path.join(output_path, f"{count}_{img_name}")
            shutil.copy(img_path, dst_path)
            count += 1

    print(f"{output_path} → Collected {count} images")


def main():
    print("Loading model...")
    model = load_model(MODEL_PATH)

    if model is None:
        print("Model loading failed.")
        return

    create_dirs()

    print("\nProcessing NORMAL images (expect False)...")
    process_folder(
        NORMAL_DIR,
        expected_label=False,
        output_path=OUTPUT_FALSE,
        model=model
    )

    print("\nProcessing POTHOLE images (expect True)...")
    process_folder(
        POTHOLE_DIR,
        expected_label=True,
        output_path=OUTPUT_TRUE,
        model=model
    )

    print("\nDone!")


if __name__ == "__main__":
    main()