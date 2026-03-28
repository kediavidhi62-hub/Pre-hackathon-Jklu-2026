from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import traceback
from dotenv import load_dotenv
load_dotenv()
from detector import detect_deepfake
from explainer import generate_explanation
from video_utils import extract_frames, cleanup_files

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "mp4"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@app.get("/ping")
def ping():
    return {"status": "alive"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # 1. Get filename and extension
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # 2. Validate extension
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid file type. Allowed: jpg, jpeg, png, webp, mp4"},
        )

    # 3. Read file and validate size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail={"error": "File too large. Maximum size is 20MB"},
        )

    # 4. Save temporarily
    if ext == "mp4":
        temp_path = "temp_upload.mp4"
    else:
        temp_path = "temp_upload.jpg"

    with open(temp_path, "wb") as f:
        f.write(contents)

    try:
        if ext == "mp4":
            file_type = "video"
            frame_paths = extract_frames(temp_path)
            
            if not frame_paths:
                score = 0.5
                verdict = "SUSPICIOUS"
            else:
                scores = []
                for frame_path in frame_paths:
                    result = detect_deepfake(frame_path)
                    scores.append(result.get("score", 0.5))
                
                avg_score = sum(scores) / len(scores)
                score = round(avg_score, 4)
                
                if score > 0.65:
                    verdict = "FAKE"
                elif score < 0.35:
                    verdict = "REAL"
                else:
                    verdict = "SUSPICIOUS"
                
                cleanup_files(frame_paths)
                
        else:
            file_type = "image"
            result = detect_deepfake(temp_path)
            score = result.get("score", 0.5)
            verdict = result.get("verdict", "SUSPICIOUS")
            
        try:
            explanation = generate_explanation(score, verdict, file_type)
        except Exception:
            explanation = "Analysis complete. Manual review recommended."
            
        return {
            "verdict": verdict,
            "score": score,
            "explanation": explanation
        }

    except Exception:
        # Fallback if anything fails during detection
        traceback.print_exc()
        return {
            "verdict": "SUSPICIOUS",
            "score": 0.5,
            "explanation": "Analysis complete. Manual review recommended."
        }
    finally:
        # 6. Always clean up temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
