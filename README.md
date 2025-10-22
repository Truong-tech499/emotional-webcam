<<<<<<< HEAD
# emotional-webcam
=======
# Real-time Emotion Recognition

A PyTorch-based project for real-time facial emotion recognition using convolutional neural networks (CNN).

## Model Files

```
# Real-time Emotion Recognition

A PyTorch-based project for real-time facial emotion recognition using convolutional neural networks (CNN).

## Model Files

- Best model: `emotion_cnn.pth` - Model với accuracy cao nhất
- Latest training state: `emotion_cnn.pth.tar` - Trạng thái training gần nhất
- Model backups: `backups/models/emotion_cnn.pth.backup_*` - Các bản backup của model tốt

## Training Visualization

Project sử dụng TensorBoard để theo dõi quá trình training với các metrics:

### Metrics cơ bản:
- Training/Validation Loss
- Training/Validation Accuracy
- Learning Rate
- Memory Usage

### Metrics nâng cao:
- Confusion Matrix
- Per-class Accuracy
- Gradient & Weight Distributions
- Sample Predictions
- Model Architecture Graph

### Cách sử dụng TensorBoard:
```bash
# Start TensorBoard server
tensorboard --logdir=runs/emotion_train

# Mở trình duyệt tại địa chỉ
http://localhost:6006
```

Hoặc trong Jupyter Notebook:
```python
%load_ext tensorboard
%tensorboard --logdir=runs/emotion_train
```

### Training Logs
- Logs được lưu trong thư mục `runs/emotion_train/`
- Mỗi lần train tạo một log file mới
- Có thể so sánh nhiều lần train khác nhau

## Features

- Real-time face detection using MediaPipe or Haar cascades
- Emotion classification into 7 categories: angry, disgusted, fearful, happy, neutral, sad, surprised
- GPU acceleration support
- TensorBoard integration for training visualization
- Modular code structure for easy customization

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Truong_Tech499/emotion-recognition.git
cd emotion-recognition
```

2. Create conda environment (recommended):
```bash
conda env create -f environment.yml
conda activate emotion
```

Or install requirements with pip:
```bash
pip install -r requirements.txt
```

## Dataset Structure

Organize your dataset in the following structure:
```
data/
├── train/
│   ├── angry/
│   ├── disgusted/
│   ├── fearful/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprised/
├── val/
│   └── ...
└── test/
    └── ...
```

## Training

To train the model:

```bash
python train.py
```

Training parameters can be modified in `train.py`. The best model will be saved as `emotion_cnn.pth`.

Monitor training progress with TensorBoard:
```bash
tensorboard --logdir=runs
```

## Real-time Detection

To run real-time emotion detection using your webcam:

```bash
python detect.py
```

Press 'q' to quit.

## Configuration

Key parameters can be modified in the respective scripts:

- `model.py`: CNN architecture
- `train.py`: Training hyperparameters
- `detect.py`: Face detection method, confidence thresholds

## Dependencies

- Python 3.7+
- PyTorch
- OpenCV
- MediaPipe (optional, recommended for face detection)
- TensorBoard
- NumPy
- Pillow

## Code Structure

```
src/
├── model.py      # CNN architecture
├── train.py      # Training pipeline
├── detect.py     # Real-time detection
└── utils.py      # Helper functions
```

## Performance

The model achieves:
- Training accuracy: ~90%
- Validation accuracy: ~85%
- Real-time inference: 15-30 FPS (depends on hardware)

## Contributing

Feel free to submit issues, fork the repository and create pull requests for any improvements.
