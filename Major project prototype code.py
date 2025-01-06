import os
import numpy as np
import librosa
import librosa.display
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Step 1: Dataset Generation

def generate_audio_dataset(directory, label_type, num_samples=50, sample_rate=22050, duration=2):
    """
    Generate synthetic audio dataset.
    :param directory: Target directory to save audio files.
    :param label_type: 'distress' or 'non_distress' to define audio characteristics.
    :param num_samples: Number of audio files to generate.
    :param sample_rate: Sampling rate of the audio signal.
    :param duration: Duration of each audio file in seconds.
    """
    for i in range(num_samples):
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        if label_type == "distress":
            audio_signal = 0.5 * np.sin(2 * np.pi * 1000 * t)  # High frequency
        else:
            audio_signal = 0.5 * np.sin(2 * np.pi * 300 * t)  # Low frequency
        
        audio_signal += 0.01 * np.random.randn(len(audio_signal))  # Add noise
        filename = os.path.join(directory, f"{label_type}_{i}.wav")
        write(filename, sample_rate, (audio_signal * 32767).astype(np.int16))

base_path = "./audio_dataset"
os.makedirs(f"{base_path}/distress", exist_ok=True)
os.makedirs(f"{base_path}/non_distress", exist_ok=True)

# Generate synthetic audio datasets
generate_audio_dataset(f"{base_path}/distress", "distress")
generate_audio_dataset(f"{base_path}/non_distress", "non_distress")

# Step 2: Data Preprocessing

def load_audio_files(directory, label):
    data, labels = [], []
    for file in os.listdir(directory):
        if file.endswith('.wav'):
            filepath = os.path.join(directory, file)
            audio, sr = librosa.load(filepath, sr=None)
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64, fmax=8000)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            data.append(mel_spec_db)
            labels.append(label)
    return data, labels

# Load distress and non-distress audio
distress_data, distress_labels = load_audio_files(f"{base_path}/distress", 1)
non_distress_data, non_distress_labels = load_audio_files(f"{base_path}/non_distress", 0)

# Combine and shuffle data
all_data = distress_data + non_distress_data
all_labels = distress_labels + non_distress_labels

combined = list(zip(all_data, all_labels))
np.random.shuffle(combined)
all_data, all_labels = zip(*combined)

# Convert to numpy arrays
X = np.array([np.expand_dims(data, axis=-1) for data in all_data])
y = np.array(all_labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Building

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 4: Training the Model

history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)

# Step 5: Evaluation

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Step 6: Visualization of Results

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
