import streamlit as st
import os
from PIL import Image
import time
from utils import load_model, preprocess_image, predict_pothole, get_severity

# Page Configuration
st.set_page_config(
    page_title="Pothole Detection System",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MODEL_PATH = "model.h5"

# Custom CSS for UI Enhancements
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .main .block-container {
        padding-top: 2rem;
    }
    .stProgress .st-bo {
        background-color: #f63366;
    }
    .severity-low {
        color: #ffc107;
        font-weight: bold;
    }
    .severity-medium {
        color: #fd7e14;
        font-weight: bold;
    }
    .severity-severe {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State for History
if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.title("ℹ️ Project Info")
        st.write("Detecting potholes from road images using Deep Learning to improve road safety and maintenance.")
        
        st.divider()
        st.subheader("📊 Dataset Info")
        st.write("The model was trained on a comprehensive dataset containing images of normal roads and roads with varying degrees of potholes.")
        
        st.divider()
        st.subheader("🧠 Model Info")
        st.write("- **Architecture**: Convolutional Neural Network (CNN)")
        st.write("- **Framework**: TensorFlow/Keras")
        st.write("- **Classes**: Pothole, Normal Road")
        
        st.divider()
        st.subheader("📝 Instructions")
        st.markdown('''
        1. Upload a road image or use the camera to take a photo.
        2. Click **Predict** to analyze the image.
        3. Review the prediction, confidence score, and severity level.
        ''')

    # --- Main Content ---
    st.title("🛣️ Pothole Detection System")
    st.markdown("""
    Welcome to the Pothole Detection System. Upload an image of a road, and the deep learning model will analyze it to determine if there is a pothole present.
    """)

    # Try to load model
    model = get_model()
    if model is None:
        st.error(f"⚠️ Model file `{MODEL_PATH}` not found. Please ensure the model exists in the project directory.")
        
    # --- Real World Applications ---
    with st.expander("🌍 Real-World Applications", expanded=False):
        st.markdown("""
        - **Smart City Infrastructure**: Automate the monitoring of road conditions and integrate with city dashbaords.
        - **Road Maintenance**: Prioritize repair work efficiently based on the severity levels detected.
        - **Autonomous Vehicles**: Enhance navigation systems to identify and avoid road hazards dynamically.
        """)

    st.divider()

    # Input Section
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📸 Input Image")
        input_type = st.radio("Choose input method:", ["Upload Image", "Camera Capture"], horizontal=True)
        
        uploaded_file = None
        if input_type == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        else:
            uploaded_file = st.camera_input("Take a picture")

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Input Image", use_column_width=True)
            
            # Predict Button
            predict_btn = st.button("🔍 Predict", use_container_width=True, type="primary")

    with col2:
        st.subheader("🎯 Prediction Result")
        
        if uploaded_file is not None and predict_btn:
            if model is None:
                st.warning("Cannot predict without a valid model.")
            else:
                with st.spinner("Analyzing image..."):
                    try:
                        # Preprocess
                        img_batch = preprocess_image(image)
                        
                        # Predict
                        confidence = predict_pothole(model, img_batch)
                        
                        # Determine class and confidence
                        is_pothole = confidence >= 0.5
                        final_prob = confidence if is_pothole else 1.0 - confidence
                        label = "Pothole Detected" if is_pothole else "Normal Road"
                        
                        # Container for result visually
                        result_container = st.container(border=True)
                        with result_container:
                            if is_pothole:
                                st.markdown(f"### ⚠️ {label}")
                                st.error("A pothole has been detected in this image. Please drive safely.")
                                
                                # Severity
                                severity = get_severity(final_prob)
                                if severity == "Low":
                                    st.info(f"**Severity Level:** {severity}")
                                elif severity == "Medium":
                                    st.warning(f"**Severity Level:** {severity}")
                                else:
                                    st.error(f"**Severity Level:** {severity}")
                            else:
                                st.markdown(f"### ✅ {label}")
                                st.success("The road appears to be clear of potholes. Have a safe journey!")
                                severity = "N/A"
                                
                            # Confidence Bar
                            st.markdown(f"**Confidence Score:** {final_prob:.2%}")
                            st.progress(float(final_prob))
                            
                        # Save to history
                        st.session_state.history.append({
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "label": label,
                            "confidence": final_prob,
                            "severity": severity if is_pothole else "N/A"
                        })
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
        elif uploaded_file is None:
            st.info("Upload an image and click Predict to see results.")

    # --- History Section ---
    if len(st.session_state.history) > 0:
        st.divider()
        st.subheader("🕒 Prediction History")
        
        # Display history in a cleanly formatted way securely
        history_header1, history_header2, history_header3, history_header4 = st.columns(4)
        history_header1.markdown("**Time**")
        history_header2.markdown("**Result**")
        history_header3.markdown("**Confidence**")
        history_header4.markdown("**Severity**")
        
        from copy import copy
        for entry in reversed(st.session_state.history):
            history_col1, history_col2, history_col3, history_col4 = st.columns(4)
            with history_col1:
                st.write(entry["time"])
            with history_col2:
                if "Pothole" in entry["label"]:
                    st.markdown(f"<span style='color:#dc3545; font-weight:bold;'>{entry['label']}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='color:#28a745; font-weight:bold;'>{entry['label']}</span>", unsafe_allow_html=True)
            with history_col3:
                st.write(f"{entry['confidence']:.2%}")
            with history_col4:
                if entry['severity'] == "Low":
                    st.markdown(f"<span class='severity-low'>{entry['severity']}</span>", unsafe_allow_html=True)
                elif entry['severity'] == "Medium":
                    st.markdown(f"<span class='severity-medium'>{entry['severity']}</span>", unsafe_allow_html=True)
                elif entry['severity'] == "Severe":
                    st.markdown(f"<span class='severity-severe'>{entry['severity']}</span>", unsafe_allow_html=True)
                else:
                    st.write(entry["severity"])

if __name__ == "__main__":
    main()
