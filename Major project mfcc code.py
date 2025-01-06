import os
import json
import math
import numpy as np
import librosa
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Paths and Constants
DATASET_PATH = "./UrbanSound8K"
JSON_PATH = "./data.json"
MODEL_PATH = "./sos_detection_model.h5"
SAMPLE_RATE = 22050
DURATION = 3  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# Preprocessing Function
def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    label_mapping = {}
    label_index = 0

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath != dataset_path:
            label = dirpath.split(os.sep)[-1]
            if label not in label_mapping:
                label_mapping[label] = label_index
                label_index += 1

            data["mapping"].append(label)
            print(f"Processing {label}")

            for f in filenames:
                file_path = os.path.join(dirpath, f)
                try:
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                    for s in range(num_segments):
                        start_sample = num_samples_per_segment * s
                        finish_sample = start_sample + num_samples_per_segment

                        mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample], sr=sr, n_fft=n_fft,
                                                    n_mfcc=n_mfcc, hop_length=hop_length).T

                        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(label_mapping[label])
                            print(f"Processed segment {s+1} for {file_path}")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

# Model Definition
def build_model(input_shape, num_classes):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Plot Training History
def plot_history(history):
    fig, axs = plt.subplots(2)

    axs[0].plot(history.history['accuracy'], label='train accuracy')
    axs[0].plot(history.history['val_accuracy'], label='validation accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()
    axs[0].set_title('Model Accuracy')

    axs[1].plot(history.history['loss'], label='train loss')
    axs[1].plot(history.history['val_loss'], label='validation loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()
    axs[1].set_title('Model Loss')

    plt.show()

if __name__ == "__main__":
    # Preprocess Data
    save_mfcc(DATASET_PATH, JSON_PATH)

    # Load Data
    with open(JSON_PATH, "r") as fp:
        data = json.load(fp)

    if not data["mfcc"] or not data["labels"]:
        raise ValueError("The dataset is empty. Please ensure the dataset path and files are correct.")

    X = np.array(data["mfcc"], dtype=object)
    y = np.array(data["labels"], dtype=int)

    if X.size == 0 or y.size == 0:
        raise ValueError("The processed data arrays are empty. Please check the preprocessing step.")

    # Ensure consistent input shapes
    X = np.array([np.array(xi) for xi in X], dtype=float)

    # Validate and remap labels
    unique_labels = np.unique(y)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_mapping[label] for label in y], dtype=int)

    # Get the number of unique classes
    num_classes = len(unique_labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Build and Train Model
    model = build_model((X.shape[1], X.shape[2]), num_classes)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

    # Save Model
    model.save(MODEL_PATH)

    # Plot Training History
    plot_history(history)

    # Evaluate Model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.2f}")
