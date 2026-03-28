import os
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from detector import detect_deepfake
from explainer import generate_explanation
from video_utils import extract_frames, cleanup_files

app = FastAPI(title="DeepGuard", description="AI-powered deepfake detection API")

# Allow any frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Allowed file types and max size
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "mp4"}
VIDEO_EXTENSIONS = {"mp4"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@app.get("/ping")
def ping():
    """Health check — tells you the server is running."""
    return {"status": "alive"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Upload an image or video to detect deepfakes.
    Returns verdict, score, explanation, and number of frames analyzed.
    """

    # Step 1: Validate file type
    filename = file.filename or "unknown"
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    print(f"\n[API] Received file: {filename} (type: {extension})")

    if extension not in ALLOWED_EXTENSIONS:
        print(f"[API] ERROR: Invalid file type: {extension}")
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Step 2: Read and validate file size
    file_bytes = await file.read()
    file_size = len(file_bytes)
    print(f"[API] File size: {file_size} bytes ({file_size / 1024 / 1024:.1f} MB)")

    if file_size > MAX_FILE_SIZE:
        print(f"[API] ERROR: File too large")
        raise HTTPException(status_code=400, detail="File too large. Max 20MB")

    # Step 3: Save file temporarily
    temp_filename = f"temp_{uuid.uuid4().hex[:8]}.{extension}"
    with open(temp_filename, "wb") as f:
        f.write(file_bytes)
    print(f"[API] Saved temp file: {temp_filename}")

    # Track all temp files for cleanup
    temp_files = [temp_filename]

    try:
        is_video = extension in VIDEO_EXTENSIONS
        file_type = "video" if is_video else "image"

        if is_video:
            # Step 5: VIDEO — extract frames and analyze each one
            print(f"[API] Processing VIDEO...")
            frame_paths = extract_frames(temp_filename)
            temp_files.extend(frame_paths)

            if not frame_paths:
                raise HTTPException(status_code=400, detail="Could not extract frames from video")

            # Run detection on each frame
            scores = []
            for i, frame_path in enumerate(frame_paths):
                print(f"[API] Analyzing frame {i + 1}/{len(frame_paths)}...")
                result = detect_deepfake(frame_path)
                scores.append(result["score"])

            # Average all scores
            avg_score = round(sum(scores) / len(scores), 4)
            frames_analyzed = len(frame_paths)
            print(f"[API] Average score across {frames_analyzed} frames: {avg_score}")

        else:
            # Step 6: IMAGE — analyze directly
            print(f"[API] Processing IMAGE...")
            result = detect_deepfake(temp_filename)
            avg_score = result["score"]
            frames_analyzed = 1
            print(f"[API] Score: {avg_score}")

        # Step 7: Determine final verdict
        if avg_score > 0.65:
            verdict = "FAKE"
        elif avg_score < 0.35:
            verdict = "REAL"
        else:
            verdict = "SUSPICIOUS"
        print(f"[API] Final verdict: {verdict}")

        # Step 8: Generate AI explanation
        print(f"[API] Generating explanation...")
        explanation = generate_explanation(avg_score, verdict, file_type)

        # Step 10: Return response
        response = {
            "verdict": verdict,
            "score": avg_score,
            "explanation": explanation,
            "frames_analyzed": frames_analyzed,
        }
        print(f"[API] Response: {response}")
        return response

    finally:
        # Step 9: ALWAYS clean up temp files, even if error occurs
        print(f"[API] Cleaning up {len(temp_files)} temp files...")
        cleanup_files(temp_files)
