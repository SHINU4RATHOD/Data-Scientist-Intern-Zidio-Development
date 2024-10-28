import streamlit as st
import numpy as np
import librosa
import librosa.display
from keras.models import load_model
import matplotlib.pyplot as plt

# Load your saved LSTM model
model = load_model("mymodel.keras")

# Custom CSS for background image and styling
st.markdown(
    """
    <style>
    body {
        background-image: url('file:///C:/Users/SHINU%20RATHOD/Desktop/internship%20assignment/05_Zidio%20Development/1_Speech%20Emotion%20Recognition(SER)/01_Dataset/SERpycode/bimg1.webp');  /* Ensure correct path format */
        background-size: cover;  /* Cover the entire screen */
        color: white;  /* Text color */
        font-family: 'Roboto', sans-serif;
    }
    h1 {
        text-align: center;
        margin-top: 20px;
    }
    .input-container {
        margin: 20px;
        padding: 20px;
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.8); /* Semi-transparent background */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .prediction {
        margin-top: 20px;
        padding: 20px;
        border-radius: 10px;
        background-color: rgba(10, 128, 10, 0.4); /* Green background */
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to extract MFCC features from the uploaded audio
def extract_mfcc_from_audio(audio_data, sample_rate):
    mfcc = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
    return mfcc

# Function to predict emotion from the extracted MFCCs
def predict_emotion(mfccs):
    mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension
    mfccs = np.expand_dims(mfccs, axis=-1)  # Add channel dimension for LSTM input
    prediction = model.predict(mfccs)
    emotion = np.argmax(prediction)
    return emotion

# Create a dictionary to map numerical predictions to emotion labels
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Streamlit UI
st.title("Speech Emotion Recognition App")

# File uploader in Streamlit for audio input
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Load the uploaded audio file
    st.audio(uploaded_file, format="audio/wav")
    
    # Convert the audio file to waveform and sample rate using librosa
    audio_data, sample_rate = librosa.load(uploaded_file, sr=None)
    
    # Display waveform plot
    st.write("Audio Waveform:")
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(audio_data, sr=sample_rate)
    st.pyplot(fig)
    
    # Extract MFCCs from the audio
    mfccs = extract_mfcc_from_audio(audio_data, sample_rate)
    
    # Predict emotion
    emotion = predict_emotion(mfccs)
  
    # Display the predicted emotion in a fancy way
    st.markdown(
        f"""
        <div class="prediction">
            The detected emotion is: {emotion_labels[emotion]}
        </div>
        """,
        unsafe_allow_html=True
    )
