import streamlit as st
# Use a flag to control page config
def set_page_configuration():
    st.set_page_config(layout="wide")
from feature import *
from Webfunc import *
import soundfile as sf
import io
import numpy as np
import librosa
import streamlit as st

sample_rate = 22050
set_page_configuration()
# Define the mapping from gender-specific labels to generic emotion labels

# Define class labels
class_labels = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

from streamlit_navigation_bar import st_navbar

with st.sidebar:
    st.page_link(
        "Web.py",
        label = "Home",
        icon=":material/home:"
    )
    

# Create a sample layout with columns
col1, col2, col3 = st.columns([1, 2, 1])  # Adjust ratios for left, center, and right columns

with col1:
    page_bg_img = """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("");
        background-size: cover;
    }
    [data-testid="stHeader"] {
        
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)


with col2:
        # Streamlit app
    st.title("Speech Emotion Recognition")

    # Display an image
    st.image("https://images.unsplash.com/photo-1561446289-4112a4f79116?q=80&w=1374&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", caption="Speech Emotion Recognition")

    st.markdown("<h1 style='font-size:32px;'>Upload an audio file (WAV format) to predict.</h1>", unsafe_allow_html=True)

    # File upload
    st.markdown("<p style='font-size:24px;'><strong>Choose an audio file:</strong></p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["wav"])

    # Input for recording duration
    st.markdown("<p style='font-size:24px;'><strong>Select recording duration (seconds):</strong></p>", unsafe_allow_html=True)
    duration = st.slider("", min_value=1, max_value=120, value=10, step=3)


        # Record button
    if st.button("Record Audio"):
        # Record audio for the selected duration
        temp_file_path = record_audio(duration=duration)
        
        # Display the recorded audio
        st.audio(temp_file_path, format='audio/wav')
        
        # Provide a download link for the recorded file
        with open(temp_file_path, 'rb') as file:
            st.download_button(label="Download Recorded Audio", data=file, file_name="recorded_audio.wav")
        
        # Add a header below the download button
        st.markdown("### Prediction")
        # Add any additional content or analysis below
        st.write("Please select the model you would like to make prediction on.")

    # Streamlit app
    if uploaded_file is not None:
        audio_data, sr = sf.read(io.BytesIO(uploaded_file.read()))
        st.audio(uploaded_file, format='audio/wav')
        # Confirmation message
        st.markdown("<p style='font-size:26px; font-weight:bold;'>Audio file successfully uploaded and played!</p>", unsafe_allow_html=True)

        # Model selection radio button
        st.markdown("<p style='font-size:26px; font-weight:bold;'>Select a model:</p>", unsafe_allow_html=True)
        frequency_model = st.radio(
            label="",
            options=["XGBoost", "CNN", "LGBM", "VGGNET", "RESNET", "DENSENET", "Ensemble(LGBM+XGB)"]
        )


        if st.button("Predict"):
            temp_file_path = "temp_audio.wav"
            with open(temp_file_path, 'wb') as f:
                f.write(uploaded_file.read())

            # Check if audio length is longer than 5 seconds
            duration = librosa.get_duration(filename=temp_file_path)

            # Example usage (assuming frequency_model is set to one of the options)
            if frequency_model == "XGBoost":
                emotion_percentages = predict_with_xgboost(temp_file_path)
                display_emotions(emotion_percentages)
            elif frequency_model == "CNN":
                emotion_percentages = predict_with_cnn(temp_file_path)
                display_emotions(emotion_percentages)
            elif frequency_model == "LGBM":
                emotion_percentages = predict_with_lgbm(temp_file_path)
                display_emotions(emotion_percentages)
            elif frequency_model == "VGGNET":
                emotion_percentages = predict_with_vggnet(temp_file_path)
                display_emotions(emotion_percentages)
            elif frequency_model == "RESNET":
                emotion_percentages = predict_with_resnet(temp_file_path)
                display_emotions(emotion_percentages)
            elif frequency_model == "DENSENET":
                emotion_percentages = predict_with_densenet(temp_file_path)
                display_emotions(emotion_percentages)
            elif frequency_model == "Ensemble(LGBM+XGB)":
                emotion_percentages = predict_with_ensemble(temp_file_path)
                display_emotions(emotion_percentages)
            else:
                st.error("Selected model is not supported.")

# Optional: Plot the results if needed

with col3:
    page_bg_img = """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.unsplash.com/photo-1568014533879-1c6fd0860c18?q=80&w=1364&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
    }
    [data-testid="stHeader"] {
        
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)







