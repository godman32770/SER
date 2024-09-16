from Web import *
from feature import *
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import streamlit as st

# Function to record audio with adjustable duration
def record_audio(duration=10, sample_rate=44100):
    # Query the default input device
    input_device = sd.default.device[0]
    device_info = sd.query_devices(input_device, 'input')
    max_input_channels = device_info['max_input_channels']

    # Use mono if the device supports only 1 channel, otherwise use stereo
    channels = 1 if max_input_channels == 1 else 2

    st.write(f"Recording for {duration} seconds with {channels} channel(s)...")
    
    # Record the audio
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
    sd.wait()  # Wait until recording is finished
    st.write("Recording complete.")
    
    # Save to temporary file
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    wav.write(temp_wav.name, sample_rate, recording)
    
    return temp_wav.name
