import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import os

@tf.keras.utils.register_keras_serializable()
def load_model(model_path):
    """
    Loads the trained Keras model from the given path.
    """
    if not os.path.exists(model_path):
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(image, target_size=(128, 128)):
    """
    Preprocesses the PIL Image to match the expected model input shape.
    Converts to numpy array, resizes, normalizes, and expands dimensions.
    """

    # Convert PIL image to numpy array
    img_array = np.array(image)

    # Handle image channels
    if img_array.shape[-1] == 4:  # RGBA → RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    elif len(img_array.shape) == 2:  # Grayscale → RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

    # Resize image to model input size
    img_resized = cv2.resize(img_array, target_size)

    # Normalize pixel values
    img_normalized = img_resized.astype("float32") / 255.0

    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)

    return img_batch

def predict_pothole(model, image_batch):
    """
    Runs the model prediction on the processed image batch.
    Returns the probability of the image being a pothole.
    Assuming binary classification where higher probability -> Pothole.
    """
    prediction = model.predict(image_batch)
    
    # Check model output shape to handle different binary classification styles
    if len(prediction[0]) > 1:
        # One-hot encoded output, e.g., [Normal_Prob, Pothole_Prob]
        prob = float(prediction[0][1])
    else:
        # Single sigmoid output
        prob = float(prediction[0][0])
        
    return prob

def get_severity(confidence):
    """
    Determines pothole severity based on the model's confidence.
    """
    if confidence < 0.65:
        return "Low"
    elif confidence < 0.85:
        return "Medium"
    else:
        return "Severe"
