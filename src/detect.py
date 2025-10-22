import os
import time
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageOps
import numpy as np
from datetime import datetime
from collections import OrderedDict, deque, Counter
from typing import List, Tuple, Optional, Dict

# Initialize face detection globals
_HAS_MEDIAPIPE = False
_DETECTOR_BACKEND = "cascade"  # Use Haar Cascade by default
mp_face_detector = None  # Global variable for MediaPipe detector

# Try to import MediaPipe if available
try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
    _DETECTOR_BACKEND = "mediapipe"
    mp_face_detector = mp.solutions.face_detection.FaceDetection(
        min_detection_confidence=0.5,
        model_selection=0  # 0=short range, 1=full range
    )
    print("[INFO] Using MediaPipe for face detection")
except ImportError:
    print("[INFO] MediaPipe not available, using Haar Cascade for face detection")

# ---------- Config ----------
import argparse

# Paths
MODEL_PATH = r"D:\nam 4\emotion_cnn.pth"
IMAGE_PATH = None  # None để dùng webcam
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
REINFORCE_DIR = "reinforce_images"  # Directory for saving misclassified images
os.makedirs(REINFORCE_DIR, exist_ok=True)

# Device & Model Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
TOP_K = 3  # show top-K emotions
BATCH_SIZE = 4  # batch size for TTA inference
USE_TTA = True  # Test Time Augmentation
FPS = 30  # target FPS for video processing

# Face Tracking
SMOOTH_WINDOW = 15  # frames for emotion smoothing
MAX_FACE_AGE = 10  # frames before removing lost face
MAX_FACE_DIST = 100  # max pixels between frames for same face
TARGET_FPS = 30  # target FPS for webcam

# Visualization
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.9
THICKNESS = 2
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
# -----------------------------


# === 1️⃣ Định nghĩa model (giống file train.py) ===
class EmotionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# === 2️⃣ Hàm xoá prefix trong checkpoint ===
def _strip_prefix(state_dict, prefix="model."):
    new_state = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            k = k[len(prefix):]
        new_state[k] = v
    return new_state


# === 3️⃣ Hàm load checkpoint an toàn ===
def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt or "classes" not in ckpt:
        raise RuntimeError("❌ Checkpoint format không đúng! Hãy kiểm tra lại file .pth được tạo từ train.py")

    classes = ckpt["classes"]
    state = ckpt["model_state"]


    # Tạo lại model đúng cấu trúc
    model = EmotionModel(num_classes=len(classes))
    model.load_state_dict(state, strict=True)
    model.eval()
    print("[INFO] ✅ Model loaded successfully!")
    return model.to(device), classes


# === 4️⃣ Chuẩn bị transform ảnh ===
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def apply_clahe_to_bgr(face_bgr):
    """Apply CLAHE to improve contrast on a BGR numpy array and return PIL RGB image."""
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb)


def predict_emotion_tta(
    model: nn.Module,
    pil_img: Image.Image,
    classes: List[str],
    top_k: int = 3,
    use_tta: bool = True,
    batch_size: int = BATCH_SIZE
) -> List[Tuple[str, float]]:
    """Predict emotion with Test Time Augmentation.
    
    Args:
        model: Trained emotion model
        pil_img: Input image
        classes: List of class names
        top_k: Number of top predictions to return
        use_tta: Whether to use Test Time Augmentation
        batch_size: Batch size for TTA inference
        
    Returns:
        List of (emotion, confidence) tuples
    """
    model.eval()
    
    # Create augmented versions
    imgs = [pil_img]
    if use_tta:
        imgs.append(ImageOps.mirror(pil_img))  # horizontal flip
        
    # Process in batches
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i:i + batch_size]
            batch_tensor = torch.stack([transform(img) for img in batch])
            batch_tensor = batch_tensor.to(DEVICE)
            
            outputs = model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())

    # Average probabilities
    avg_probs = torch.cat(all_probs).mean(dim=0)
    
    # Get top-k predictions
    values, indices = torch.topk(avg_probs, k=top_k)
    return [(classes[idx], val.item()) for idx, val in zip(indices, values)]


def detect_faces(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect faces in frame using available backend.
    
    Args:
        frame: Input BGR frame
        
    Returns:
        List of (x, y, width, height) face rectangles
    """
    faces = []
    
    if _HAS_MEDIAPIPE and mp_face_detector is not None:
        # MediaPipe detection (faster & more accurate)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_detector.process(rgb)
        
        if results and results.detections:
            h, w = frame.shape[:2]
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Clip to image bounds
                x = max(0, x)
                y = max(0, y)
                width = min(w - x, width)
                height = min(h - y, height)
                
                faces.append((x, y, width, height))
    else:
        # Haar cascade fallback
        if not hasattr(detect_faces, "cascade"):
            cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
            if cascade.empty():
                raise RuntimeError(f"Cannot load cascade from: {FACE_CASCADE_PATH}")
            detect_faces.cascade = cascade

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces.cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5
        )
    
    return faces


# === 6️⃣ Main: test ảnh hoặc webcam với face detection ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to image file (optional)")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--smooth", type=int, default=SMOOTH_WINDOW, help="Smoothing window size")
    parser.add_argument("--detector", type=str, default=("mediapipe" if _HAS_MEDIAPIPE else "haar"), choices=["mediapipe", "haar"], help="Face detector backend to use")
    args = parser.parse_args()

    # Override configs from CLI args
    image_path = args.image or IMAGE_PATH
    smooth_window = args.smooth

    print("[INFO] Loading models...")
    model, classes = load_checkpoint(MODEL_PATH, device=DEVICE)
    detector_backend = args.detector
    if detector_backend == "mediapipe":
        if not _HAS_MEDIAPIPE:
            print("[WARN] MediaPipe not installed; falling back to Haarcascade")
            detector_backend = "haar"
        else:
            mp_face = mp.solutions.face_detection
            mp_face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    # Haarcascade (fallback)
    if detector_backend == "haar":
        face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        if face_cascade.empty():
            raise RuntimeError(f"❌ Cannot load face cascade from: {FACE_CASCADE_PATH}")
    print("[INFO] Models loaded successfully! Detector=", detector_backend)
    
    # Process single image
    if image_path and os.path.exists(image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Cannot read image: {image_path}")
            return
            
        faces = detect_faces(frame)
        
        if len(faces) == 0:
            print("[RESULT] No faces detected!")
            cv2.putText(
                frame,
                "No faces detected!",
                (20, 40),
                FONT,
                FONT_SCALE,
                RED,
                THICKNESS
            )
        
        # Process each face
        for (x, y, w, h) in faces:
            # Get and preprocess face
            face_crop = frame[y:y+h, x:x+w]
            face_clahe = apply_clahe_to_bgr(face_crop)
            
            # Predict with TTA
            predictions = predict_emotion_tta(
                model,
                face_clahe,
                classes,
                top_k=TOP_K,
                use_tta=USE_TTA
            )
            
            # Draw results
            confidence = predictions[0][1]  # top-1 confidence
            thickness = max(1, min(3, int(confidence * 4)))
            
            # Draw bbox
            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                GREEN,
                thickness
            )
            
            # Draw labels with background
            for i, (emotion, prob) in enumerate(predictions):
                label = f"{emotion} ({prob:.0%})"
                text_size = cv2.getTextSize(
                    label,
                    FONT,
                    FONT_SCALE,
                    THICKNESS
                )[0]
                
                # Background
                cv2.rectangle(
                    frame,
                    (x, y-text_size[1]-10-i*30),
                    (x+text_size[0], y-5-i*30),
                    BLACK,
                    -1
                )
                
                # Text
                cv2.putText(
                    frame,
                    label,
                    (x, y-10-i*30),
                    FONT,
                    FONT_SCALE,
                    GREEN,
                    THICKNESS
                )
        
        cv2.imshow("Emotion Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # --- Realtime webcam ---
    print(f"[INFO] Starting webcam (index={args.cam})... Press 'q' to quit.")
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"❌ Cannot open camera {args.cam}")
        return

    # smoothing buffer for each face (mapping face_id -> data)
    face_data = {}  # face_id -> (center_pos, predictions)
    next_face_id = 1
    
    # tracking parameters
    max_face_age = 10  # frames before removing lost face
    max_distance = 100  # pixels between frames to consider same face
    target_fps = 30
    frame_time = 1.0 / target_fps

    # Measure actual FPS
    fps_buffer = deque(maxlen=30)
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break

        # Calculate actual FPS
        current_time = time.time()
        fps_buffer.append(1.0 / (current_time - last_time + 1e-6))
        last_time = current_time
        avg_fps = sum(fps_buffer) / len(fps_buffer)

        # Mirror frame for more natural interaction
        frame = cv2.flip(frame, 1)

        # Detect faces
        faces = detect_faces(frame)

        if len(faces) == 0:
            cv2.putText(frame, "No faces detected!", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # update face age
        for face_id in list(face_data.keys()):
            face_data[face_id]['age'] += 1
            if face_data[face_id]['age'] > max_face_age:
                del face_data[face_id]
        
        # track and update faces
        current_faces = []
        
        for (x, y, w, h) in faces:
            center = (x + w//2, y + h//2)
            
            # find closest tracked face
            min_dist = float('inf')
            matched_id = None
            
            for face_id, data in face_data.items():
                dist = ((center[0] - data['center'][0])**2 + 
                       (center[1] - data['center'][1])**2)**0.5
                if dist < min_dist and dist < max_distance:
                    min_dist = dist
                    matched_id = face_id
            
            # assign new ID if no match
            if matched_id is None:
                matched_id = next_face_id
                next_face_id = (next_face_id % 10) + 1  # cycle through 1-10
                face_data[matched_id] = {
                    'predictions': deque(maxlen=smooth_window),
                    'center': center,
                    'age': 0
                }
            
            # update tracked face
            face_data[matched_id]['center'] = center
            face_data[matched_id]['age'] = 0
            current_faces.append(matched_id)
            
            # get face region and predict
            face_rgb = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_rgb, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            
            # preprocess with CLAHE and predict with TTA
            face_clahe = apply_clahe_to_bgr(frame[y:y+h, x:x+w])
            predictions = predict_emotion_tta(model, face_clahe, classes, top_k=TOP_K, use_tta=USE_TTA)
            curr_emotion, curr_prob = predictions[0]

            # update prediction history (store full prediction tuple)
            face_data[matched_id]['predictions'].append((curr_emotion, curr_prob))

            # compute smoothed emotion using weighted sum over history
            pred_queue = face_data[matched_id]['predictions']
            if len(pred_queue) > 0:
                emotion_scores = {}
                for emotion, prob in pred_queue:
                    emotion_scores[emotion] = emotion_scores.get(emotion, 0) + prob
                smoothed_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            else:
                smoothed_emotion = curr_emotion

            # draw bbox with thickness based on probability
            thickness = max(1, min(3, int(curr_prob * 4)))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness)

            # draw single label with smoothed emotion (no Face ID)
            label = f"{smoothed_emotion}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            
            # background for better visibility
            cv2.rectangle(frame, 
                         (x, y-text_size[1]-10), 
                         (x+text_size[0], y-5),
                         (0, 0, 0), -1)
            
            # draw text
            cv2.putText(frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # cleanup: remove faces not in current frame and not already deleted
        for face_id in list(face_data.keys()):
            if face_id not in current_faces and face_id in face_data:
                del face_data[face_id]
        
        # Display frame
        cv2.imshow("Emotion Detection (Press 'q' to quit, 's' to save)", frame)
        
        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and len(faces) > 0:
            # Save each detected face
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for i, (x, y, w, h) in enumerate(faces):
                face_img = frame[y:y+h, x:x+w]
                # Get emotion for this face
                face_id = current_faces[i] if i < len(current_faces) else None
                if face_id and face_id in face_data:
                    emotion = face_data[face_id]['predictions'][-1][0]  # get latest prediction
                    prob = face_data[face_id]['predictions'][-1][1]
                    save_path = os.path.join(REINFORCE_DIR, f"{timestamp}_face{i}_{emotion}_{prob:.2f}.jpg")
                    cv2.imwrite(save_path, face_img)
                    print(f"[SAVED] Face {i} ({emotion}: {prob:.0%}) to {save_path}")
            
        time.sleep(1.0 / FPS)

    cap.release()
    cv2.destroyAllWindows()
    # cleanup mediapipe
    if 'mp_face_detector' in globals():
        try:
            mp_face_detector.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        # Set OpenMP environment variable
        if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Detection stopped by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
    finally:
        cv2.destroyAllWindows()
