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

1. Clone repository và cài đặt môi trường:
```powershell
# Clone repository
git clone https://github.com/Truong_Tech499/emotion-recognition.git
cd emotion-recognition

# Tạo môi trường conda (recommended)
conda env create -f environment.yml
conda activate emotion_env

# Hoặc sử dụng pip
pip install -r requirements.txt
```

2. Download face detection model:
```powershell
# Tạo thư mục models
mkdir models

# Download Haar Cascade model
curl -o models/haarcascade_frontalface_default.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
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

## Training & Detection

### Training Model

```powershell
# Kích hoạt môi trường
conda activate emotion_env

# Set biến môi trường để tránh lỗi OpenMP
$env:KMP_DUPLICATE_LIB_OK='TRUE'

# Train model (đường dẫn tương đối từ root project)
python src/train.py

# Theo dõi quá trình training với TensorBoard
tensorboard --logdir=runs/emotion_train
```

### Inference/Detection

```powershell
# Kích hoạt môi trường
conda activate emotion_env

# Set biến môi trường
$env:KMP_DUPLICATE_LIB_OK='TRUE'

# Real-time detection với webcam
python src/detect.py

# Detection với ảnh có sẵn
python src/detect.py --image path/to/image.jpg

# Các tham số khác
python src/detect.py --help
```

Các tham số cho detect.py:
- `--image`: Đường dẫn đến ảnh cần phân tích (mặc định: sử dụng webcam)
- `--cam`: Index của camera (mặc định: 0)
- `--smooth`: Kích thước cửa sổ làm mượt dự đoán (mặc định: 15 frames)
- `--detector`: Backend phát hiện khuôn mặt ("mediapipe" hoặc "haar", mặc định: mediapipe nếu có)

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
