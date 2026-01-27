
import os
from typing import Any, Optional

import cv2
import numpy as np

# Lazy imports for heavy deps (loaded on first use)
_yolo_model: Any = None
_insightface_app: Any = None
_arcface_model: Any = None


def _models_dir() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "models")


def get_yolo_face_path() -> str:
    return os.path.join(_models_dir(), "yolov8n-face.pt")


def get_detector():
    """Load YOLOv8-face detector once. Cached in module (or use @st.cache_resource in app)."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        path = get_yolo_face_path()
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"YOLO model not found at {path}. Place yolov8n-face.pt in the models/ folder."
            )
        _yolo_model = YOLO(path)
    return _yolo_model


def get_arcface_model():
    """Load InsightFace FaceAnalysis (buffalo_l) and return recognition model. Cached."""
    global _insightface_app, _arcface_model
    if _arcface_model is None:
        from insightface.app import FaceAnalysis
        _insightface_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _insightface_app.prepare(ctx_id=1, det_size=(640, 640))
        _arcface_model = _insightface_app.models["recognition"]
    return _arcface_model


def get_embedding(face_image: np.ndarray, arcface_model: Optional[Any] = None) -> Optional[np.ndarray]:
    if arcface_model is None:
        arcface_model = get_arcface_model()
    if not isinstance(face_image, np.ndarray):
        return None
    img = face_image
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = cv2.resize(img, (112, 112))
    try:
        embedding = arcface_model.get_feat(img)
        embedding = np.asarray(embedding).flatten().astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm < 1e-8:
            return None
        return (embedding / norm).astype(np.float32)
    except Exception:
        return None


def recognize_face(
    query_emb: Optional[np.ndarray],
    db_emb: np.ndarray,
    db_labels: list[str],
    threshold: float = 0.38,
) -> tuple[str, float]:
    if query_emb is None or db_emb is None or len(db_emb) == 0:
        return "Unknown", 0.0
    query_emb = np.asarray(query_emb, dtype=np.float32).flatten()
    if db_emb.dtype != np.float32:
        db_emb = np.asarray(db_emb, dtype=np.float32)
    # Cosine similarity (embeddings assumed normalized)
    sims = np.dot(db_emb, query_emb)
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    if best_score >= threshold:
        return db_labels[best_idx], best_score
    return "Unknown", best_score


def process_frame(
    bgr_image: np.ndarray,
    detector: Any,
    arcface_model: Any,
    db_emb: np.ndarray,
    db_labels: list[str],
    name_map: dict[str, str],
    threshold: float = 0.38,
) -> tuple[np.ndarray, list[tuple[str, float]]]:
    image_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    image_out = bgr_image.copy()
    results = detector(image_rgb, verbose=False)
    if not results or len(results) == 0:
        return image_out, []
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return image_out, []
    recognized: list[tuple[str, float]] = []
    for box in result.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        face_crop_rgb = image_rgb[y1:y2, x1:x2]
        if face_crop_rgb.size == 0:
            continue
        query_emb = get_embedding(face_crop_rgb, arcface_model)
        user_id, score = recognize_face(query_emb, db_emb, db_labels, threshold)
        recognized.append((user_id, score))
        color = (0, 255, 0) if user_id != "Unknown" else (0, 0, 255)
        user_id = name_map.get(user_id, user_id) if user_id != "Unknown" else user_id
        cv2.rectangle(image_out, (x1, y1), (x2, y2), color, 2)
        label_text = f"{user_id} ({score:.2f})"
        (t_w, t_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image_out, (x1, y1 - 25), (x1 + t_w, y1), color, -1)
        cv2.putText(
            image_out, label_text, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
    return image_out, recognized


def extract_largest_face(image_bgr: np.ndarray, detector: Any) -> Optional[np.ndarray]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = detector(image_rgb, verbose=False)
    if not results or len(results) == 0 or results[0].boxes is None:
        return None
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    if len(boxes) == 0:
        return None
    # Largest by area
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = int(np.argmax(areas))
    x1, y1, x2, y2 = boxes[idx]
    face = image_rgb[y1:y2, x1:x2]
    if face.size == 0:
        return None
    face = cv2.resize(face, (112, 112))
    return face
