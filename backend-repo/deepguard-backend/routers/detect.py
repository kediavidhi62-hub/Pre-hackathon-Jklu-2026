from fastapi import APIRouter, UploadFile, File, HTTPException
from utils.image_processing import validate_image, preprocess_image
from services.detector import detect_deepfake

router = APIRouter()


@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Upload an image to check if it is a deepfake.
    Accepts: jpg, jpeg, png, webp (max 10MB).
    Returns: prediction result with label and confidence.
    """
    validation = await validate_image(file)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail=validation["error"])

    contents = await file.read()
    processed_image = preprocess_image(contents)
    result = await detect_deepfake(processed_image, file.filename or "image.jpg")

    if not result["success"]:
        raise HTTPException(status_code=502, detail=result["error"])

    return {
        "success": True,
        "filename": file.filename,
        "result": result["result"],
    }
