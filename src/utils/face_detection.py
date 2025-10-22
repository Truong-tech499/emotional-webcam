"""Face detection utilities."""
import cv2
import numpy as np
from typing import Tuple, Optional

def init_face_detector(backend: str = "cascade") -> any:
    """Initialize face detector."""
    if backend == "cascade":
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        return cv2.CascadeClassifier(cascade_path)
    elif backend == "mediapipe":
        import mediapipe as mp
        return mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5)
    else:
        raise ValueError(f"Unsupported detector backend: {backend}")

def detect_face(image: np.ndarray, detector: any, backend: str = "cascade") -> Optional[Tuple[int, int, int, int]]:
    """Detect face in image and return bounding box."""
    if backend == "cascade":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            return (x, y, w, h)
    elif backend == "mediapipe":
        results = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x = int(bbox.xmin * iw)
            y = int(bbox.ymin * ih)
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)
            return (x, y, w, h)
    return None