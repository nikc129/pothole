# Pothole Detection System 🛣️

A Deep Learning-based application to identify potholes from road images. This project aims to improve road safety and maintenance monitoring by automating the detection of road hazards using computer vision techniques. 

The project consists of two main parts:
1. **Model Training & Research (`.ipynb`)**
2. **Real-time Web UI (`app.py`)**

---

## 🏗️ Project Structure
```text
project/
│
├── Pothole_Detection_Final_Project.ipynb  # Jupyter Notebook for Data Analysis & Model Training
├── model.h5                               # The exported trained deep learning model
├── app.py                                 # The main Streamlit web application
├── utils.py                               # Helper functions (Image preprocessing, predictions)
├── requirements.txt                       # Python dependencies
└── assets/                                # Directory for static assets and examples
```

---

## 1. How It Works (Jupyter Notebook)
The file `Pothole_Detection_Final_Project.ipynb` is the heart of the machine learning model creation. In this notebook, we handle:

- **Data Loading:** Pull in a dataset consisting of images categorized as "Pothole" and "Normal Road".
- **Exploratory Data Analysis (EDA):** Visualizing the data distribution and checking image dimensions to understand the problem space.
- **Data Preprocessing & Augmentation:** Applying transformations like rotations, scaling, and horizontal flips to artificially increase the size of our dataset, making the model more robust.
- **Model Architecture (CNN):** Building a Convolutional Neural Network (CNN) architecture using TensorFlow/Keras capable of binary image classification.
- **Training & Evaluation:** Training the model over multiple epochs, validating it against testing/holdout data, and generating accuracy/loss plots.
- **Model Exporting:** Once satisfactory accuracy is achieved, the model's weights and architecture are saved locally to `model.h5`.

---

## 2. Web Application (Streamlit)
The `app.py` script leverages Streamlit to provide a sleek, modern, and user-friendly interface. 

### Key Features
- **File Upload & Camera Input:** Users can either upload existing road images (`jpg`, `png`) or seamlessly switch to using their webcam/camera to snap real-time pictures.
- **Binary Classification:** Processes the image and feeds it into our loaded `model.h5`, producing a "Pothole Detected" or "Normal Road" result.
- **Confidence Scores & Severity Levels:** Visually represents the model's confidence logic (using progress bars) and maps it to a severity level (Low, Medium, Severe).
- **History Tracker:** Maintains an active state of prediction history spanning the current browser session.

---

## ⚙️ Installation

To set up the application environment on your machine, follow these steps:

1. **Clone/Navigate to the Directory:**
   ```bash
   cd path/to/pothole/folder
   ```

2. **(Optional) Create a virtual environment:**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate 
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   Install the required Python modules from the `requirements.txt` file.
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

### Step 1: Run the App
With the dependencies installed and ensuring `model.h5` exists in the folder, spin up the Streamlit server:
```bash
streamlit run app.py
```

### Step 2: Access the Application
The terminal will output a secure local URL (usually `http://localhost:8501`).
1. Navigate to this URL in your web browser.
2. The UI will appear automatically. 
3. Try uploading a road image to see the model detect a pothole!

---

**Note:** If you want to retrain the model or test a different architecture, open the `Pothole_Detection_Final_Project.ipynb` notebook and run all cells sequentially. Once your new model has finished training, save the new weights as `model.h5` inside the same parent directory and restart the Streamlit server.
