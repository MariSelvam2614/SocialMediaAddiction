import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# Paths (match YOUR structure)
# -----------------------------
DATA_DIR = "data"
MODEL_DIR = "models"

X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.csv")
Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "lstm_scaler.pkl")

SEQUENCE_LENGTH = 10

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
X_train = pd.read_csv(X_TRAIN_PATH).values
y_train = pd.read_csv(Y_TRAIN_PATH).values.ravel()

# -----------------------------
# Scale features
# -----------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, SCALER_PATH)

# -----------------------------
# Create sequences
# -----------------------------
def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y[i + seq_len])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_train, SEQUENCE_LENGTH)

# -----------------------------
# Build LSTM
# -----------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# Train
# -----------------------------
model.fit(
    X_seq,
    y_seq,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
    verbose=1
)

model.save(MODEL_PATH)

print("LSTM model trained and saved.")