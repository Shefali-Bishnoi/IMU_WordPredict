import numpy as np

from pathlib import Path

from config import (
    index_to_label,
    NORM_STATS_PATH,
    MODEL_SEQ_LEN,
)

RAW_DATA_DIR = Path(r"D:\BTP_Marker_Project\IMU_WordPredict_BTP\Dataset")

from preprocessing.io import (
    load_raw_file,
    get_sensor_matrix,
    get_char_label,
)

from preprocessing.clean import clean_sensor_matrix
from preprocessing.segment import preprocess
from inference.realtime import CharacterRecognizer


# ---------------------------------------------------------
# EXACT RAW FILE
# ---------------------------------------------------------

filepath = (
    RAW_DATA_DIR
    / "capital letters"
    / "X"
    / "S01"
    / "X-02.txt"
)

print("RAW FILE:")
print(filepath)


# ---------------------------------------------------------
# LOAD RAW
# ---------------------------------------------------------

raw = load_raw_file(filepath)

print("\nRaw shape:")
print(raw.shape)


# ---------------------------------------------------------
# EXTRACT SENSOR DATA
# ---------------------------------------------------------

sensor = get_sensor_matrix(raw)

print("\nSensor shape:")
print(sensor.shape)
print("\nRaw shape:", raw.shape)
print("First 3 raw sensor rows:")
print(sensor[:3])

print("\nLast 3 raw sensor rows:")
print(sensor[-3:])

# ---------------------------------------------------------
# CLEAN
# ---------------------------------------------------------

cleaned = clean_sensor_matrix(sensor)

print("\nCleaned shape:")
print(cleaned.shape)


# ---------------------------------------------------------
# LOAD REALTIME RECOGNIZER
# ---------------------------------------------------------

recognizer = CharacterRecognizer(arch="tcn")


# ---------------------------------------------------------
# SAME PREPROCESSING AS REALTIME
# ---------------------------------------------------------

processed = preprocess(
    cleaned,
    mean=recognizer.mean,
    std=recognizer.std,
    window_len=recognizer.seq_len,
)

print("\nProcessed shape:")
print(processed.shape)
print("\nProcessed first 3 rows:")
print(processed[:3])

print("\nProcessed last 3 rows:")
print(processed[-3:])

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

x = processed[np.newaxis, ...]

probs = recognizer.model.predict(
    x,
    verbose=0
)[0]

indices = np.argsort(probs)[::-1]


# ---------------------------------------------------------
# RESULT
# ---------------------------------------------------------

print("\n========================================")
print("TOP 10")
print("========================================")

for rank, idx in enumerate(indices[:10], start=1):
    print(
        f"{rank:2d}. "
        f"{index_to_label(int(idx)):>2s} "
        f"{probs[idx]:.8f}"
    )

print("\nPredicted:")
print(index_to_label(int(indices[0])))
print(float(probs[indices[0]]))