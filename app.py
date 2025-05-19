import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import time
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Arabic Sign Language Recognition",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
    }
    .arabic-text {
        font-family: 'Arial', sans-serif;
        direction: rtl;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

# Application title
st.markdown("<h1 class='main-header'>Arabic Sign Language Recognition</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='sub-header'>Recognize Arabic alphabet sign language using computer vision</h3>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/username/arabic-sign-language/main/logo.png", width=100, use_column_width=True)
    st.title("Options")
    app_mode = st.selectbox("Choose Mode", ["Home", "Live Detection", "Upload Image", "About"])
    
    st.markdown("---")
    st.markdown("### Arabic Alphabet")
    st.markdown("<p class='arabic-text'>أ ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ي</p>", unsafe_allow_html=True)

# Load the pre-trained model (placeholder - you'll need to replace with your actual model)
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('arabic_sign_model.h5')
        return model
    except:
        st.warning("⚠️ Model not found. Using placeholder functionality.")
        return None

# Arabic alphabet labels
arabic_alphabet = ["أ", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", 
                  "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف", 
                  "ق", "ك", "ل", "م", "ن", "ه", "و", "ي"]

# Function to preprocess the image
def preprocess_image(img):
    # Resize to the input size expected by your model
    img = cv2.resize(img, (224, 224))
    # Normalize pixel values
    img = img / 255.0
    return img

# Function to predict the sign
def predict_sign(img, model):
    if model is None:
        # Placeholder prediction logic when model isn't available
        time.sleep(1)  # Simulate processing time
        prediction = np.random.rand(28)
        prediction = prediction / np.sum(prediction)  # Normalize to sum to 1
        return prediction
    
    # Preprocess the image
    processed_img = preprocess_image(img)
    # Add batch dimension
    processed_img = np.expand_dims(processed_img, axis=0)
    # Make prediction
    prediction = model.predict(processed_img)
    return prediction[0]

# Home page
def home_page():
    st.markdown("""
    # Welcome to Arabic Sign Language Recognition
    
    This application helps recognize Arabic alphabet signs using computer vision and machine learning.
    
    ## Features:
    - **Live Detection**: Use your webcam to detect signs in real-time
    - **Upload Image**: Upload an image to recognize the sign
    - **Comprehensive Recognition**: Supports all 28 Arabic alphabet letters
    
    ## How to use:
    1. Select a mode from the sidebar
    2. For live detection, ensure your webcam is enabled
    3. For image upload, provide a clear image showing the hand sign
    
    ## About the Project:
    This project aims to bridge communication gaps by providing a tool that recognizes Arabic sign language, making digital communication more accessible.
    """)

    # Display sample signs
    st.markdown("### Sample Signs")
    cols = st.columns(4)
    for i, col in enumerate(cols):
        if i < 4:  # Just show 4 sample signs
            col.image(f"https://raw.githubusercontent.com/username/arabic-sign-language/main/sample_{i+1}.png", 
                    caption=f"Arabic letter: {arabic_alphabet[i]}", use_column_width=True)

# Live detection page
def live_detection_page():
    st.markdown("## Live Arabic Sign Language Detection")
    st.markdown("Position your hand in front of the camera showing an Arabic letter sign.")
    
    model = load_model()
    
    # Create a placeholder for the webcam feed
    video_placeholder = st.empty()
    
    # Create a placeholder for the prediction results
    results_placeholder = st.empty()
    
    # Start/Stop button
    run = st.button("Start/Stop Camera")
    
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
        
    if run:
        st.session_state.camera_running = not st.session_state.camera_running
    
    if st.session_state.camera_running:
        # This would actually use OpenCV to access the webcam in a deployed app
        # Since we can't access webcam directly in this environment, we'll simulate it
        st.markdown("⚠️ Webcam simulation active - in actual deployment, this would show your camera feed")
        
        # Simulation loop
        for _ in range(20):  # Simulate 20 frames
            if not st.session_state.camera_running:
                break
                
            # Generate a random frame for simulation
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Display the frame
            video_placeholder.image(frame, channels="BGR", use_column_width=True)
            
            # Make a prediction
            prediction = predict_sign(frame, model)
            predicted_index = np.argmax(prediction)
            predicted_letter = arabic_alphabet[predicted_index]
            
            # Display results
            col1, col2 = results_placeholder.columns(2)
            col1.markdown(f"### Predicted Letter: {predicted_letter}")
            
            # Create bar chart for top 5 predictions
            fig, ax = plt.subplots(figsize=(10, 4))
            top_indices = np.argsort(prediction)[-5:][::-1]
            top_probs = prediction[top_indices]
            top_letters = [arabic_alphabet[i] for i in top_indices]
            
            ax.barh(top_letters, top_probs)
            ax.set_xlabel('Probability')
            ax.set_title('Top 5 Predictions')
            
            col2.pyplot(fig)
            
            time.sleep(0.1)  # Simulate frame rate
    else:
        st.markdown("Click 'Start/Stop Camera' to begin live detection")

# Upload image page
def upload_image_page():
    st.markdown("## Analyze Arabic Sign Language from Image")
    
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Make prediction
        with st.spinner("Analyzing image..."):
            model = load_model()
            prediction = predict_sign(img_array, model)
            predicted_index = np.argmax(prediction)
            predicted_letter = arabic_alphabet[predicted_index]
        
        # Display results
        st.success(f"Analysis complete!")
        
        col1, col2 = st.columns(2)
        
        col1.markdown(f"### Predicted Letter: {predicted_letter}")
        col1.markdown(f"### Confidence: {prediction[predicted_index]*100:.2f}%")
        
        # Create bar chart for top 5 predictions
        fig, ax = plt.subplots(figsize=(10, 6))
        top_indices = np.argsort(prediction)[-5:][::-1]
        top_probs = prediction[top_indices]
        top_letters = [arabic_alphabet[i] for i in top_indices]
        
        ax.barh(top_letters, top_probs)
        ax.set_xlabel('Probability')
        ax.set_title('Top 5 Predictions')
        
        col2.pyplot(fig)

# About page
def about_page():
    st.markdown("""
    ## About This Project
    
    The Arabic Sign Language Recognition project aims to bridge communication gaps between the deaf community and others by leveraging computer vision and machine learning technologies.
    
    ### Technology Stack:
    - **Frontend**: Streamlit
    - **Computer Vision**: OpenCV
    - **Machine Learning**: TensorFlow, Keras
    - **Data Processing**: NumPy, Pandas
    
    ### Dataset:
    The model was trained on an RGB Arabic Alphabets Sign Language Dataset, featuring thousands of images representing the 28 letters of the Arabic alphabet.
    
    ### Model Architecture:
    The neural network uses a convolutional architecture optimized for hand gesture recognition, with transfer learning applied from pre-trained models to improve accuracy.
    
    ### Future Improvements:
    - Adding support for Arabic word recognition
    - Implementing real-time translation to text and speech
    - Mobile application development
    - Expanding the dataset with more diverse signers
    
    ### Contact:
    For questions, suggestions, or collaboration opportunities, please contact us at [example@email.com](mailto:example@email.com).
    """)

# Main application logic
if app_mode == "Home":
    home_page()
elif app_mode == "Live Detection":
    live_detection_page()
elif app_mode == "Upload Image":
    upload_image_page()
elif app_mode == "About":
    about_page()

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ for the Arabic sign language community | © 2025")
