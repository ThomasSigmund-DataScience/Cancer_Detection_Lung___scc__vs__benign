import pickle
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# COMMANDS TO USE STREAMLIT
# pip install streamlit pillow numpy tensorflow
# streamlit run cancer_detection_app.py

# Load the trained model from the same folder as the script
model_path = 'model___STRATEGY_4___40_epochs___4_convLayers___2024-11-29_185639.sav'

with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)

# Streamlit app UI
st.title('Lung Tissue Classification:')
st.subheader('Cancer scc  vs.  benign tissue')

# Upload the image
image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Check if an image is uploaded
if image_file is not None:
    # Open and display the image
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image", width=300)  # Set width to approximately 10 cm

    # Preprocess the image (resize to match model input)
    image = image.resize((150, 150))  # Adjust this size based on your model's input size
    image_array = np.array(image)

    # Expand dimensions to match the input shape (batch size, height, width, channels)
    image_array = np.expand_dims(image_array, axis=0)

    # Make the prediction
    prediction = loaded_model.predict(image_array)

    # Print the prediction probabilities for debugging
    st.write(f"Prediction Probabilities: {prediction}")
    print(f"Prediction Probabilities: {prediction}")

    # Get the class with the highest probability
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Print the predicted class for debugging
    st.write(f"Predicted Class: {predicted_class}")
    print(f"Predicted Class: {predicted_class}")

    # Display the prediction result
    if predicted_class == 0:  # 0 is for lung cancer scc
        st.write("Prediction: Cancerous")
    else:                     # 1 is for benign tissue
        st.write("Prediction: Benign")
