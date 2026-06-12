# Speech Emotion Recognition (SER)

This repository contains a Streamlit-based speech emotion recognition application that lets users upload or record an audio file and predict the dominant emotion from the speech signal.

## What this project does

- Records or uploads WAV audio
- Extracts speech features using librosa
- Runs emotion prediction with multiple model options
- Displays emotion percentages and visualizations in the browser

## Main files

- Web.py
  - Main Streamlit app interface
  - Handles file upload, recording, model selection, and prediction display

- Webfunc.py
  - Model-loading and prediction helper functions
  - Includes support for XGBoost, LGBM, CNN, VGGNet, ResNet, DenseNet, and ensemble models

- feature.py
  - Core feature extraction and preprocessing logic
  - Includes audio segmentation, MFCC-based feature generation, and emotion visualization helpers

- speech-emotion-recognition (1).ipynb
  - Experimental / training notebook for the speech emotion recognition workflow

- requirements.txt
  - Python dependencies needed to run the project

- version.py
  - Version-related utility information

## Tech stack

- Python
- Streamlit
- librosa
- NumPy
- TensorFlow / Keras
- XGBoost
- joblib
- Plotly / Matplotlib

## Setup

1. Create and activate a virtual environment
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies
   ```powershell
   pip install -r requirements.txt
   ```

3. Run the app
   ```powershell
   streamlit run Web.py
   ```

## Notes

- The app expects WAV audio input.
- Some model files are downloaded from Google Drive during runtime, so internet access may be required on first use.
- The project is focused on speech emotion recognition and uses audio feature extraction plus multiple machine learning/deep learning models.
