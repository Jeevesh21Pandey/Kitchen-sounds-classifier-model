# Pressure Cooker Whistle & Frying Sound Detection using CNNs

A lightweight audio classification system for detecting:

- **Pressure cooker whistle**
- **Frying / boiling sounds**
- **Background / non-cooking sounds**

using **Mel spectrograms + Convolutional Neural Networks (CNNs)**.

The project is designed for:

- Raspberry Pi deployment
- Real-time kitchen monitoring
- Embedded audio inference
- ONNX Runtime inference in C++

---

# Features

- Mel spectrogram based audio preprocessing
- CNN-based whistle detection
- CNN-based frying sound detection
- ONNX export support
- Raspberry Pi compatible
- TensorFlow / Keras training pipeline
- C++ real-time inference support
- Audio augmentation for robustness
- Early stopping and validation tracking

---

# Project Structure

```text
.
├── Train/
│   ├── whistle/
│   ├── frying/
│   └── not_whistle/
│
├── Test/
│   ├── whistle/
│   ├── frying/
│   └── not_whistle/
│
├── whistle_model.py
├── frying_model.py
├── predict.cpp
├── requirements.txt
└── README.md
```

---

# Dataset Structure

Training and testing datasets must follow this structure:

```text
Train/
   whistle/
   frying/
   not_whistle/

Test/
   whistle/
   frying/
   not_whistle/
```

Each folder should contain audio clips such as:

```text
.wav
.mp3
.flac
.ogg
.m4a
```

---

# Audio Processing Pipeline

The system uses the following preprocessing chain:

```text
Audio
   ↓
Mel Spectrogram
   ↓
Log Scaling (dB)
   ↓
Normalization
   ↓
CNN
```

---

# Spectrogram Parameters

| Parameter | Value |
|---|---|
| Sample Rate | 48 kHz |
| FFT Window | 25 ms |
| Hop Length | 10 ms |
| Mel Bins | 128 |
| Clip Duration | 3 seconds |

---

# Model Architecture

The CNN architecture:

```text
Conv2D(32)
↓
MaxPooling
↓
Conv2D(64)
↓
MaxPooling
↓
Conv2D(64)
↓
MaxPooling
↓
GlobalAveragePooling
↓
Dense(32)
↓
Dropout
↓
Sigmoid Output
```

---

# Training

## Install dependencies

```bash
pip install tensorflow librosa numpy scikit-learn tf2onnx onnx
```

---

# Train Whistle Detector

```bash
python whistle_model.py train
```

Optional arguments:

```bash
python whistle_model.py train \
    --epochs 20 \
    --batch-size 16 \
    --val-split 0.2
```

---

# Train Frying Detector

```bash
python frying_model.py train
```

---

# Prediction

## Predict using trained model

```bash
python whistle_model.py predict model.h5 audio.wav
```

Example:

```bash
python whistle_model.py predict whistle_model.h5 test.wav
```

---

# ONNX Export

The training script supports ONNX export:

```bash
python whistle_model.py train \
    --onnx-output whistle_model.onnx
```

---

# Raspberry Pi Deployment

The project is optimized for deployment on:

- Raspberry Pi 4
- Raspberry Pi 5

using:

- ONNX Runtime
- C++ inference pipeline
- FFTW
- FFmpeg

---

# C++ Inference Pipeline

The `predict.cpp` implementation performs:

```text
Audio Loading
↓
FFT
↓
Mel Filterbank
↓
Log Spectrogram
↓
Normalization
↓
ONNX Runtime Inference
```

---

# Compile C++ Inference

Example compilation:

```bash
g++ predict.cpp -o predict \
    -lonnxruntime \
    -lfftw3f \
    -lavformat \
    -lavcodec \
    -lavutil \
    -lswresample
```

---

# Run Inference

```bash
./predict audio.wav
```

---

# Current Performance

Approximate performance achieved with current datasets:

| Task | Accuracy |
|---|---|
| Whistle Detection | ~80–85% |
| Frying Detection | ~70–80% |

Performance depends heavily on:

- microphone quality
- room acoustics
- dataset diversity
- background noise

---

# Known Limitations

- Small datasets can lead to overfitting
- Similar high-frequency sounds may cause false positives
- Different kitchens/microphones may require retraining
- ONNX conversion may occasionally fail with certain TensorFlow graph versions

---

# Future Improvements

Planned upgrades:

- Sliding window training
- Real-time streaming inference
- Hierarchical classifier architecture
- CNN + temporal modeling
- Multi-class unified detector
- Quantized ONNX models
- Noise-robust feature extraction

---

# Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- ONNX Runtime
- FFTW
- FFmpeg
- C++

---

# Example Applications

- Smart kitchen monitoring
- Cooker whistle detection
- Cooking activity recognition
- Embedded audio AI systems
- IoT-based kitchen automation

---

# Author

Developed as an embedded audio AI project for real-time cooking sound recognition and Raspberry Pi deployment.
