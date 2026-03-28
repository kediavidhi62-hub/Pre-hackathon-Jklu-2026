import os
import httpx

MODEL_API_URL = os.getenv("MODEL_API_URL", "")


async def detect_deepfake(image_bytes: bytes, filename: str) -> dict:
    """
    Send image bytes to the deployed Render model API and return the prediction.
    Returns a dict with 'label' (real/fake) and 'confidence' score.
    """
    if not MODEL_API_URL:
        return {
            "success": False,
            "error": "MODEL_API_URL is not configured in .env",
        }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            files = {"file": (filename, image_bytes, "image/jpeg")}
            response = await client.post(MODEL_API_URL, files=files)
            response.raise_for_status()
            result = response.json()

        return {
            "success": True,
            "result": result,
        }

    except httpx.TimeoutException:
        return {
            "success": False,
            "error": "Model API request timed out. Please try again.",
        }
    except httpx.HTTPStatusError as e:
        return {
            "success": False,
            "error": f"Model API returned error: {e.response.status_code}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to connect to model API: {str(e)}",
        }
