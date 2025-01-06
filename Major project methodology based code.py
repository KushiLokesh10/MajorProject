import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load and preprocess audio data
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    
    # Convert to spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    return log_spectrogram, mfcc

# Data augmentation (pitch shift, noise addition, time-stretching)
def augment_data(signal):
    # Pitch shift
    signal = librosa.effects.pitch_shift(signal, sr=22050, n_steps=4)
    
    # Add noise
    noise = np.random.randn(len(signal))
    signal = signal + 0.005 * noise
    
    # Time stretching
    signal = librosa.effects.time_stretch(signal, 1.1)  # stretch by 10%
    
    return signal

# Load dataset (Example for ESC-50 or UrbanSound8K)
def load_dataset(dataset_dir):
    features, labels = [], []
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.wav'):
            label = 0 if 'distress' in filename else 1  # Example: assign 0 for distress and 1 for non-distress
            file_path = os.path.join(dataset_dir, filename)
            signal, _ = librosa.load(file_path, sr=None)
            
            # Augment data for variety
            augmented_signal = augment_data(signal)
            
            # Extract features
            log_spectrogram, mfcc = extract_features(file_path)
            
            features.append(log_spectrogram)
            labels.append(label)
    
    return np.array(features), np.array(labels)

# Load and preprocess data
dataset_dir = '/path/to/your/dataset'  # Replace with your dataset path
X, y = load_dataset(dataset_dir)

# Data preparation: Reshape data for CNN input
X = np.expand_dims(X, axis=-1)  # Add channel dimension
X = X.astype('float32') / 255.0  # Normalize

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model design: CNN for binary classification
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification (Distress vs. Non-Distress)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_class = (y_pred > 0.5).astype('int32')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class)
print("Confusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_class))

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
