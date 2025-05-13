# train_model.py

import os
import librosa
import numpy as np
import kagglehub
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Download dataset
path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
print("Dataset downloaded to:", path)

# 2. Map file codes to emotions
emotion_map = {
    '01':'neutral','02':'calm','03':'happy','04':'sad',
    '05':'angry','06':'fearful','07':'disgust','08':'surprised'
}

# 3. Load audio and extract MFCCs
X, y = [], []
for root, _, files in os.walk(path):
    for fname in files:
        if fname.endswith('.wav'):
            eid = fname.split('-')[2]
            label = emotion_map.get(eid)
            if label:
                audio, sr = librosa.load(os.path.join(root, fname), sr=None)
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T
                X.append(mfcc)
                y.append(label)

# 4. Pad sequences
max_len = max(seq.shape[0] for seq in X)
X_pad = pad_sequences(X, maxlen=max_len, padding='post', dtype='float32')

# 5. Encode labels
y_enc = LabelEncoder().fit_transform(y)
y_cat = tf.keras.utils.to_categorical(y_enc)

# 6. Split
X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y_cat, test_size=0.2, random_state=42
)

# 7. Build model
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(max_len, X_pad.shape[2])),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(y_cat.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 8. Train
history = model.fit(
    X_train, y_train,
    epochs=30, batch_size=32,
    validation_data=(X_test, y_test)
)

# 9. Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.4f}")

# 10. Save
model.save('improved_emotion_model.h5')
print("Model saved as improved_emotion_model.h5")

# 11. Optional: print max_len for app.py
print("Use this MAX_LEN in your app:", max_len)
