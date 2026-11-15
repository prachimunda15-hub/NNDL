
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# =====================================================
# 1. CONFIGURATION
# =====================================================

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATASET_PATH = Path(
    "/mnt/c/Users/Prachi Singh/OneDrive/เอกสาร/Downloads/CREMA-D-WAV"
)

OUTPUT_DIR = Path.home() / "datasets" / "CREMA_D_CNN_BiLSTM"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
MAX_LEN = 200

BATCH_SIZE = 16
EPOCHS = 80

# =====================================================
# 2. EMOTION LABELS
# =====================================================

emotion_map = {
    "ANG": 0,
    "DIS": 1,
    "FEA": 2,
    "HAP": 3,
    "NEU": 4,
    "SAD": 5
}

emotion_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad"
]

NUM_CLASSES = 6

# =====================================================
# 3. FEATURE EXTRACTION
# =====================================================

def extract_features(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
        top_db=80
    )

    # Normalize between 0 and 1
    mel_db = (mel_db + 80.0) / 80.0

    mel_db = mel_db.T

    # Padding / truncation
    if mel_db.shape[0] < MAX_LEN:

        padding = MAX_LEN - mel_db.shape[0]

        mel_db = np.pad(
            mel_db,
            ((0, padding), (0, 0)),
            mode="constant"
        )

    else:
        mel_db = mel_db[:MAX_LEN, :]

    return mel_db.astype(np.float32)


# =====================================================
# 4. LOAD CREMA-D DATASET
# =====================================================

X_PATH = OUTPUT_DIR / "X.npy"
Y_PATH = OUTPUT_DIR / "y.npy"
SPEAKER_PATH = OUTPUT_DIR / "speakers.npy"

if all(p.exists() for p in [X_PATH, Y_PATH, SPEAKER_PATH]):

    print("Loading cached CREMA-D features...")

    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    speakers = np.load(SPEAKER_PATH)

else:

    print("Extracting CREMA-D features...")

    files = sorted(DATASET_PATH.rglob("*.wav"))

    print("Total WAV files:", len(files))

    valid_files = []

    for file in files:

        parts = file.stem.split("_")

        if len(parts) < 4:
            continue

        speaker_id = parts[0]
        emotion = parts[2]

        if emotion not in emotion_map:
            continue

        valid_files.append(
            (file, speaker_id, emotion)
        )

    print("Valid recordings:", len(valid_files))

    X = np.zeros(
        (len(valid_files), MAX_LEN, N_MELS, 1),
        dtype=np.float32
    )

    y = np.zeros(len(valid_files), dtype=np.int32)

    speakers = np.empty(
        len(valid_files),
        dtype="<U10"
    )

    for i, (file, speaker, emotion) in enumerate(valid_files):

        try:

            feature = extract_features(str(file))

            X[i, :, :, 0] = feature

            y[i] = emotion_map[emotion]

            speakers[i] = speaker

        except Exception as e:

            raise RuntimeError(
                f"Feature extraction failed for {file}: {e}"
            )

        if (i + 1) % 500 == 0:

            print(
                f"Processed {i + 1}/{len(valid_files)} files"
            )

    print("Feature extraction completed!")

    np.save(X_PATH, X)
    np.save(Y_PATH, y)
    np.save(SPEAKER_PATH, speakers)

    print("Features saved:", OUTPUT_DIR)

print("\nDataset shape:", X.shape)
print("Labels shape:", y.shape)
print("Unique speakers:", len(np.unique(speakers)))

# =====================================================
# 5. SPEAKER-INDEPENDENT DATA SPLIT
# =====================================================

# 15% speakers for testing

split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.15,
    random_state=SEED
)

train_val_idx, test_idx = next(
    split1.split(X, y, groups=speakers)
)

# Approximately 15% overall for validation

split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.1765,
    random_state=SEED
)

train_relative, val_relative = next(
    split2.split(
        X[train_val_idx],
        y[train_val_idx],
        groups=speakers[train_val_idx]
    )
)

train_idx = train_val_idx[train_relative]
val_idx = train_val_idx[val_relative]

print("\nTrain:", len(train_idx))
print("Validation:", len(val_idx))
print("Test:", len(test_idx))

# Verify speaker separation

train_speakers = set(speakers[train_idx])
val_speakers = set(speakers[val_idx])
test_speakers = set(speakers[test_idx])

assert train_speakers.isdisjoint(val_speakers)
assert train_speakers.isdisjoint(test_speakers)
assert val_speakers.isdisjoint(test_speakers)

print("Speaker leakage check: PASSED")

# Check all six classes are represented in training

print("Training classes:", np.unique(y[train_idx]))

if len(np.unique(y[train_idx])) != NUM_CLASSES:
    raise ValueError(
        "A training split is missing an emotion class. "
        "Change the speaker split before training."
    )

# =====================================================
# 6. PREPARE TRAINING DATA
# =====================================================

X_train = X[train_idx]
y_train = y[train_idx]

X_val = X[val_idx]
y_val = y[val_idx]

X_test = X[test_idx]
y_test = y[test_idx]

# Class weights

classes = np.arange(NUM_CLASSES)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("Class weights:", class_weights)

# =====================================================
# 7. CNN-BiLSTM MODEL
# =====================================================

def build_cnn_bilstm(input_shape):

    inputs = layers.Input(shape=input_shape)

    # CNN BLOCK 1

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D((2, 2))(x)

    # CNN BLOCK 2

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D((2, 2))(x)

    # CNN BLOCK 3

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D((2, 2))(x)

    # CNN BLOCK 4

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D((2, 2))(x)

    # Convert CNN output to sequences
    # Input: 200 x 128
    # After four pooling layers: 12 x 8 x 128

    x = layers.Reshape((12, 1024))(x)

    # BiLSTM BLOCK 1

    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.3
        )
    )(x)

    # BiLSTM BLOCK 2

    x = layers.Bidirectional(
        layers.LSTM(
            64,
            return_sequences=False,
            dropout=0.3
        )
    )(x)

    # DENSE CLASSIFIER

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="CREMA_D_CNN_BiLSTM"
    )

    return model


# =====================================================
# 8. BUILD AND COMPILE
# =====================================================

model = build_cnn_bilstm(
    input_shape=(MAX_LEN, N_MELS, 1)
)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =====================================================
# 9. CALLBACKS
# =====================================================

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),

    ModelCheckpoint(
        str(OUTPUT_DIR / "cnn_bilstm_best.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )
]

# =====================================================
# 10. TRAIN
# =====================================================

history = model.fit(
    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=EPOCHS,
    batch_size=BATCH_SIZE,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1
)

# =====================================================
# 11. SAVE MODEL
# =====================================================

model.save(
    str(OUTPUT_DIR / "cnn_bilstm_final.keras")
)

print("\nModel saved successfully!")

# =====================================================
# 12. TEST EVALUATION
# =====================================================

best_model = tf.keras.models.load_model(
    str(OUTPUT_DIR / "cnn_bilstm_best.keras")
)

test_loss, test_accuracy = best_model.evaluate(
    X_test,
    y_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

print("\nTest Accuracy:", test_accuracy)
print("Test Loss:", test_loss)

# =====================================================
# 13. PREDICTIONS
# =====================================================

y_prob = best_model.predict(
    X_test,
    batch_size=BATCH_SIZE
)

y_pred = np.argmax(y_prob, axis=1)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        labels=classes,
        target_names=emotion_names,
        zero_division=0
    )
)

print(
    "Accuracy:",
    accuracy_score(y_test, y_pred)
)

print(
    "Macro F1:",
    f1_score(y_test, y_pred, average="macro")
)

print(
    "Weighted F1:",
    f1_score(y_test, y_pred, average="weighted")
)

# =====================================================
# 14. CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=classes
)

plt.figure(figsize=(10, 8))

plt.imshow(cm, cmap="Blues")

plt.xticks(
    classes,
    emotion_names,
    rotation=45
)

plt.yticks(
    classes,
    emotion_names
)

plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")

plt.title("CREMA-D CNN-BiLSTM Confusion Matrix")

plt.colorbar()

for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    str(OUTPUT_DIR / "cnn_bilstm_confusion_matrix.png"),
    dpi=300
)

plt.show()

# =====================================================
# 15. TRAINING CURVES
# =====================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN-BiLSTM Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN-BiLSTM Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig(
    str(OUTPUT_DIR / "cnn_bilstm_training_curves.png"),
    dpi=300
)

plt.show()

print("\nTraining and evaluation completed!")
print("All outputs saved in:", OUTPUT_DIR)
