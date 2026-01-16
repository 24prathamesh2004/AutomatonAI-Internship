"""
FastAPI Backend for Tyre Segmentation
Place your best.pt file in: backend/models/best.pt
"""

import os
import io
import base64
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO

app = FastAPI(title="Tyre Segmentation API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model path - Place best.pt here: backend/models/best.pt
MODEL_PATH = Path(__file__).parent / "models" / "best.pt"
model = None


def load_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Model not found. Place best.pt at: {MODEL_PATH}"
            )
        model = YOLO(str(MODEL_PATH))
    return model


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/segment/image")
async def segment_image(file: UploadFile = File(...)):
    """Segment tyres in an image and return count with annotated image."""
    
    # Load model
    yolo = load_model()
    
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Run inference
    results = yolo(img, task="segment")[0]
    
    # Get annotated image
    annotated = results.plot()
    
    # Encode to base64
    _, buffer = cv2.imencode(".png", annotated)
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    
    # Extract detections
    detections = []
    if results.boxes is not None:
        for i, box in enumerate(results.boxes):
            detections.append({
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            })
    
    return {
        "count": len(detections),
        "image_base64": img_base64,
        "detections": detections
    }


@app.post("/segment/video")
async def segment_video(file: UploadFile = File(...)):
    """Segment and track tyres in video with counting."""
    
    # Load model
    yolo = load_model()
    
    # Save uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        contents = await file.read()
        tmp.write(contents)
        input_path = tmp.name
    
    # Output path
    output_path = tempfile.mktemp(suffix="_output.mp4")
    
    try:
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video file")
        
        # Video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_counts = []
        tracked_ids = set()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run tracking
            results = yolo.track(frame, persist=True, task="segment")[0]
            
            # Count in this frame
            frame_count = 0
            if results.boxes is not None:
                frame_count = len(results.boxes)
                
                # Track unique IDs
                if results.boxes.id is not None:
                    for track_id in results.boxes.id:
                        tracked_ids.add(int(track_id))
            
            frame_counts.append(frame_count)
            
            # Write annotated frame
            annotated = results.plot()
            out.write(annotated)
        
        cap.release()
        out.release()
        
        # Return result with video URL
        return {
            "total_count": len(tracked_ids) if tracked_ids else max(frame_counts) if frame_counts else 0,
            "video_url": f"/video/{os.path.basename(output_path)}",
            "frame_counts": frame_counts
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup input file
        if os.path.exists(input_path):
            os.unlink(input_path)


@app.get("/video/{filename}")
async def get_video(filename: str):
    """Serve processed video."""
    video_path = Path(tempfile.gettempdir()) / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path, media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

