import librosa
import numpy as np
from collections import Counter
from Web import *
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import requests
class_labels = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

label_mapping = {
    'female_angry': 'angry', 'male_angry': 'angry',
    'female_calm': 'calm', 'male_calm': 'calm',
    'female_disgust': 'disgust', 'male_disgust': 'disgust',
    'female_fear': 'fear', 'male_fear': 'fear',
    'female_happy': 'happy', 'male_happy': 'happy',
    'female_neutral': 'neutral', 'male_neutral': 'neutral',
    'female_sad': 'sad', 'male_sad': 'sad',
    'female_surprise': 'surprise', 'male_surprise': 'surprise'
}

# Assume the helper functions (zcr, mfcc, rmse, pad_or_truncate) and the extract_features function are already defined.
def pad_or_truncate(features, target_length):
    if len(features.shape) == 1:
        if features.shape[0] < target_length:
            pad_width = target_length - features.shape[0]
            features = np.pad(features, (0, pad_width), mode='constant')
        else:
            features = features[:target_length]
    elif len(features.shape) == 2:
        if features.shape[0] < target_length:
            pad_width = target_length - features.shape[0]
            features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
        else:
            features = features[:target_length, :]
    return features

#----------------------------------
def segment_audio(data, sample_rate, window_length, hop_duration):
    window_length_samples = int(window_length * sample_rate)
    hop_duration_samples = int(hop_duration * sample_rate)
    
    segments = []
    for start in range(0, len(data) - window_length_samples + 1, hop_duration_samples):
        end = start + window_length_samples
        segment = data[start:end]
        segments.append(segment)
    
    return segments

#-----------------------
def extract_features(data):
    result = np.array([])
    mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=58)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    result = np.array(mfccs_processed)
    return result


def preprocess_for_model(file_path, duration=4, offset=1, target_length=58):
    # Load the audio file
    data, sample_rate = librosa.load(file_path, sr=None)  # Load with original sample rate
    # Extract features (make sure extract_features is suitable for XGBoost)
    features = extract_features(data)
    features = pad_or_truncate(features, target_length)
    return features


#------------------------------------------------

def display_emotions(emotion_percentages):
    """
    Display emotions and their percentages on Streamlit with the highest emotion highlighted.
    
    Parameters:
    - emotion_percentages (dict): A dictionary where keys are emotions and values are their percentages.
    """
    # Find the emotion with the highest percentage
    highest_emotion = max(emotion_percentages, key=emotion_percentages.get)
    
    # Display emotions and percentages
    for emotion, percentage in emotion_percentages.items():
        if emotion == highest_emotion:
            st.markdown(f"<p style='font-size:20px; font-weight:bold;'>{emotion}: {percentage:.2f}%</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='font-size:20px;'>{emotion}: {percentage:.2f}%</p>", unsafe_allow_html=True)
    
    # Call the visualization function
    display_emotion_visualizations(emotion_percentages)

def display_emotion_visualizations(emotion_percentages):
    # Get emotions and percentages from the dictionary
    emotions = list(emotion_percentages.keys())
    percentages = list(emotion_percentages.values())
    
    # Plotly Bar Chart
    fig = go.Figure(data=[go.Bar(x=emotions, y=percentages, marker_color='skyblue')])
    fig.update_layout(
        title='Emotion Percentage Distribution',
        xaxis_title='Emotions',
        yaxis_title='Percentage'
    )
    
    # Matplotlib Pie Chart
    fig1, ax = plt.subplots()
    ax.pie(percentages, labels=emotions, autopct='%1.1f%%', colors=plt.cm.Paired(range(len(emotions))))
    ax.set_title('Emotion Percentage Distribution')

    # Display the plot in Streamlit
    st.pyplot(fig1)
    st.plotly_chart(fig)

#-----------------------------------------



def preprocess_and_predict(file_path,loaded_model, window_length=2, hop_duration=0.25, target_length=58):
    # Load the audio file
    data, sample_rate = librosa.load(file_path, sr=22050)  # Load with original sample rate
    
    # Check if the audio length is shorter than the window length
    if len(data) < window_length * sample_rate:
        # If shorter, process the entire audio as a single segment
        segments = [data]
    else:
        # Otherwise, segment the audio
        segments = segment_audio(data, sample_rate, window_length, hop_duration)
    
    # Define class labels (generic emotions)
    class_labels = list(set(label_mapping.values()))  # Unique generic labels
    # Initialize a counter for emotion occurrences
    emotion_counter = Counter()
    
    
    # Process each segment
    for segment in segments:
        # Extract features from the segment
        features = extract_features(segment)
        # Ensure that the feature vector is the correct length
        features = pad_or_truncate(features, target_length)
        # Reshape features to match the input shape expected by your model
        features = np.reshape(features, (1, features.shape[0], 1))  # Reshape for CNN input
        # Use the CNN model for inference
        prediction = loaded_model.predict(features)
        predicted_class_index = np.argmax(prediction)
        # Get the full list of class labels from your model
        model_class_labels = list(label_mapping.keys())
        predicted_label = model_class_labels[predicted_class_index]
        # Map the predicted label to the generic emotion
        predicted_emotion = label_mapping.get(predicted_label, "Unknown")
        
        # Update the emotion counter
        emotion_counter[predicted_emotion] += 1
    
    # Calculate the percentage of each emotion
    total_segments = len(segments)
    emotion_percentages = {emotion: (count / total_segments) * 100 for emotion, count in emotion_counter.items()}
    
    return emotion_percentages

def preprocess_and_predict_boosting(file_path, loaded_model, window_length=2, hop_duration=0.25, target_length=58):
    # Load the audio file
    data, sample_rate = librosa.load(file_path, sr=22050)  # Load with original sample rate
    
    # Segment the audio into smaller windows
    if len(data) < window_length * sample_rate:
        segments = [data]  # If the audio is shorter than the window length, use the entire audio
    else:
        segments = segment_audio(data, sample_rate, window_length, hop_duration)  # Otherwise, segment the audio
    
    # Define class labels (generic emotions)
    class_labels = list(set(label_mapping.values()))  # Unique generic labels
    
    # Initialize a counter for emotion occurrences
    emotion_counter = Counter()
    
    # Process each segment
    for segment in segments:
        # Extract features from the segment
        features = extract_features(segment)
        
        # Ensure that the feature vector is the correct length
        features = pad_or_truncate(features, target_length)
        features = np.reshape(features, (1, -1))  # Reshape for XGBoost or CatBoost input
        
        # Use the boosting model for inference
        prediction = loaded_model.predict(features)
        
        # Get the full list of class labels from your model
        model_class_labels = list(label_mapping.keys())
        
        # If the boosting model outputs a single class prediction (e.g., `predict` returns a class index):
        predicted_class_index = int(prediction[0])  # Ensure it's an integer index
        
        # Map the predicted label to the generic emotion
        predicted_label = model_class_labels[predicted_class_index]
        predicted_emotion = label_mapping.get(predicted_label, "Unknown")
        
        # Update the emotion counter
        emotion_counter[predicted_emotion] += 1
    
    # Calculate the percentage of each emotion
    total_segments = len(segments)
    emotion_percentages = {emotion: (count / total_segments) * 100 for emotion, count in emotion_counter.items()}
    
    return emotion_percentages


# Function to download the model with progress bar
def download_model_from_gdrive(gdrive_url, output_name):
    with st.spinner('Downloading model...'):
        # Get file size from the headers
        response = requests.get(gdrive_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        block_size = 1024  # 1 Kilobyte
        progress_bar = st.progress(0)
        downloaded_size = 0

        with open(output_name, 'wb') as f:
            for data in response.iter_content(block_size):
                f.write(data)
                downloaded_size += len(data)
                progress = downloaded_size / total_size
                progress_bar.progress(progress)

        progress_bar.empty()  # Remove the progress bar after download completes

def predict_with_xgboost(file_path):
    gdrive_url = 'https://drive.google.com/uc?id=190_XdT_4ibvnRGI_zXZCSVPbiK720qne'  # Replace with actual model file ID
    model_path = 'xgb_model_augment2_optuna_malefemale.joblib'
    download_model_from_gdrive(gdrive_url, model_path)
    xgb_model = joblib.load(model_path)
    emotion_percentages = preprocess_and_predict_boosting(file_path, xgb_model)
    return emotion_percentages

def predict_with_lgbm(file_path):
    gdrive_url = 'https://drive.google.com/uc?id=1KO_4F4XVq-62cMbvdJwgdpjOL8H_pHWC'
    model_path = 'lgbm_model_augment2_malefemale_optuna.joblib'
    download_model_from_gdrive(gdrive_url, model_path)
    lgbm_model = joblib.load(model_path)
    emotion_percentages = preprocess_and_predict_boosting(file_path, lgbm_model)
    return emotion_percentages

def predict_with_ensemble(file_path):
    gdrive_url = 'https://drive.google.com/uc?id=15qGBkf-To5HAsSW88Nyz-72kzJes65pX'
    model_path = 'ensemble_optuna.joblib'
    download_model_from_gdrive(gdrive_url, model_path)
    ensemble_model = joblib.load(model_path)
    emotion_percentages = preprocess_and_predict_boosting(file_path,ensemble_model)
    return emotion_percentages

def predict_with_cnn(file_path):
    gdrive_url = 'https://drive.google.com/uc?id=1cMfMlzSDO0LYD09k905YkfGaUOhaHgRq'
    model_path = 'cnn.joblib'
    download_model_from_gdrive(gdrive_url, model_path)
    cnn_model = joblib.load(model_path)
    emotion_percentages = preprocess_and_predict(file_path,cnn_model)
    return emotion_percentages

def predict_with_vggnet(file_path):
    gdrive_url = 'https://drive.google.com/uc?id=1ADmg6vX9NwbB7Wnfp-_s7h3UYLCbDlHq'
    model_path = 'vgg.joblib'
    download_model_from_gdrive(gdrive_url, model_path)
    vgg_model = joblib.load(model_path)
    emotion_percentages = preprocess_and_predict(file_path,vgg_model)   
    return emotion_percentages

def predict_with_resnet(file_path):
    gdrive_url = 'https://drive.google.com/uc?id=1hRUO48dCvabxiKEUSeWPJlKetdjossO1'
    model_path = 'resnet.joblib'
    download_model_from_gdrive(gdrive_url, model_path)
    resnet_model = joblib.load(model_path)
    emotion_percentages = preprocess_and_predict(file_path,resnet_model)
    return emotion_percentages

def predict_with_densenet(file_path):
    gdrive_url = 'https://drive.google.com/uc?id=1AtYZZv4YvY3YVDx6aD5IVNi69F1NqM'
    model_path = 'densenet.joblib'
    download_model_from_gdrive(gdrive_url, model_path)
    dense_model = joblib.load(model_path)
    emotion_percentages = preprocess_and_predict(file_path,dense_model)
    return emotion_percentages


