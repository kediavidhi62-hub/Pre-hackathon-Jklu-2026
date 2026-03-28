from io import BytesIO
from PIL import Image
from fastapi import UploadFile

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def get_file_extension(filename: str) -> str:
    """Extract and return the lowercase file extension."""
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()


async def validate_image(file: UploadFile) -> dict:
    """
    Validate the uploaded file:
    - Must have an allowed image extension (jpg, jpeg, png, webp)
    - Must not exceed 10MB
    Returns dict with 'valid' bool and optional 'error' message.
    """
    extension = get_file_extension(file.filename or "")
    if extension not in ALLOWED_EXTENSIONS:
        return {
            "valid": False,
            "error": f"Invalid file type '.{extension}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        }

    contents = await file.read()
    await file.seek(0)

    if len(contents) > MAX_FILE_SIZE_BYTES:
        return {
            "valid": False,
            "error": f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB.",
        }

    try:
        image = Image.open(BytesIO(contents))
        image.verify()
    except Exception:
        return {
            "valid": False,
            "error": "File is not a valid image.",
        }

    return {"valid": True}


def preprocess_image(image_bytes: bytes) -> bytes:
    """
    Open the image with Pillow, convert to RGB, and return as JPEG bytes.
    This ensures a consistent format is sent to the model API.
    """
    image = Image.open(BytesIO(image_bytes))
    image = image.convert("RGB")

    max_size = 1024
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.LANCZOS)

    output = BytesIO()
    image.save(output, format="JPEG", quality=90)
    output.seek(0)
    return output.read()
