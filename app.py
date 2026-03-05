import streamlit as st
from PIL import Image
import time
from utils import load_model, preprocess_image, predict_pothole, draw_detections

# Page Configuration
st.set_page_config(
    page_title="Pothole Detection System",
    page_icon="🛣️",
    layout="wide",
)

# Load YOLO model
@st.cache_resource
def get_model():
    return load_model()

# Session history
if "history" not in st.session_state:
    st.session_state.history = []

def main():

    # Sidebar
    with st.sidebar:
        st.title("ℹ️ Project Info")
        st.write(
            "Detect potholes in road images using a YOLOv8 deep learning model."
        )

        st.divider()

        st.subheader("🧠 Model Info")
        st.write("• Model: YOLOv8 Object Detection")
        st.write("• Framework: Ultralytics")
        st.write("• Task: Pothole Detection")

        st.divider()

        st.subheader("📌 Instructions")
        st.write(
            """
1. Upload a road image
2. Click **Predict**
3. The system will detect potholes
"""
        )

    # Main Title
    st.title("🛣️ Pothole Detection System")
    st.write(
        "Upload an image of a road and the AI model will detect potholes."
    )

    # Load model
    model = get_model()

    if model is None:
        st.error("❌ Model failed to load. Check if Yolov8-fintuned-on-potholes.pt exists.")
        st.stop()

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Input Image")

        uploaded_file = st.file_uploader(
            "Upload road image", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)

            predict_btn = st.button("🔍 Detect Potholes")

    with col2:
        st.subheader("🎯 Detection Result")

        if uploaded_file and predict_btn:

            with st.spinner("Analyzing road surface..."):

                try:

                    img_array = preprocess_image(image)

                    detections = predict_pothole(model, img_array)

                    if len(detections) == 0:
                        st.success("✅ No potholes detected")

                    else:
                        st.error(f"⚠️ {len(detections)} pothole(s) detected")

                        for d in detections:
                            st.write(
                                f"Confidence: {d['confidence']:.2f} | Severity: {d['severity']}"
                            )

                    annotated = draw_detections(img_array, detections)

                    st.image(annotated, caption="Detected Potholes")

                    # Save history
                    st.session_state.history.append(
                        {
                            "time": time.strftime("%H:%M:%S"),
                            "count": len(detections),
                        }
                    )

                except Exception as e:
                    st.error(f"Error during prediction: {e}")

    # History
    if st.session_state.history:

        st.divider()
        st.subheader("🕒 Detection History")

        for h in reversed(st.session_state.history):

            st.write(f"{h['time']} → {h['count']} potholes detected")


if __name__ == "__main__":
    main()