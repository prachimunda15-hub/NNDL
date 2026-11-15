
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Input shape
INPUT_SHAPE = (551, 128, 1)
NUM_CLASSES = 8

# Input
inputs = layers.Input(shape=INPUT_SHAPE)

# CNN Block 1
x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D((2, 2))(x)

# CNN Block 2
x = layers.Conv2D(64, (3, 3), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D((2, 2))(x)

# CNN Block 3
x = layers.Conv2D(128, (3, 3), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)

# Attention mechanism
channels = x.shape[-1]

attention = layers.GlobalAveragePooling2D()(x)
attention = layers.Dense(
    channels // 4, activation="relu"
)(attention)
attention = layers.Dense(
    channels, activation="sigmoid"
)(attention)

attention = layers.Reshape((1, 1, channels))(attention)

# Apply attention
x = layers.Multiply()([x, attention])

# Classification
x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.4)(x)

x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

# Create model
model = models.Model(inputs, outputs)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display architecture
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    ),

    ModelCheckpoint(
        "best_cnn_attention.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]

# Training
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    callbacks=callbacks,
    class_weight=class_weights
)

# Testing
test_loss, test_accuracy = model.evaluate(
    X_test, y_test
)

print("Test Accuracy:", test_accuracy)

# Save model
model.save("speech_emotion_cnn_attention.keras")
