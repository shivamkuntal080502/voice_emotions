import importlib, subprocess, sys

# --- Auto-install missing dependencies at runtime ---
for pkg in ("librosa",):
    try:
        importlib.import_module(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---- Constants ----
MODEL_PATH = 'improved_emotion_model.h5'
EMOTION_LABELS = [
    'neutral', 'calm', 'happy', 'sad',
    'angry', 'fearful', 'disgust', 'surprised'
]
MAX_LEN = 495    # ← replace with the printed max_len from train_model.py

# ---- Load resources ----
@st.cache(allow_output_mutation=True)
def load_resources():
    model = load_model(MODEL_PATH)
    le = LabelEncoder().fit(EMOTION_LABELS)
    return model, le

model, label_encoder = load_resources()

# ---- Streamlit UI ----
st.title("🎤 Speech Emotion Recognition")
st.write("Upload a WAV file and get the predicted emotion:")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])
if uploaded_file:
    # Save to disk for librosa
    with open('temp.wav', 'wb') as f:
        f.write(uploaded_file.read())

    # Extract features
    audio, sr = librosa.load('temp.wav', sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T
    mfcc_padded = pad_sequences([mfcc], maxlen=MAX_LEN, padding='post', dtype='float32')

    # Predict
    preds = model.predict(mfcc_padded)
    idx = np.argmax(preds)
    emotion = label_encoder.inverse_transform([idx])[0]
    confidence = float(preds[0][idx])

    # Display
    st.audio('temp.wav', format='audio/wav')
    st.markdown(f"**Predicted emotion:** {emotion.capitalize()}")
    st.progress(confidence)
    st.write(f"Confidence: {confidence:.2%}")
