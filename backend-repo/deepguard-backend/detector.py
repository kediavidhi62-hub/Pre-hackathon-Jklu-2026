import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load API keys from .env file
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

# Set up the HuggingFace client
client = InferenceClient(token=HF_API_KEY)

# Model that detects AI-generated vs real images (works on free tier!)
MODEL_NAME = "umm-maybe/AI-image-detector"


def detect_deepfake(image_path: str) -> dict:
    """
    Send an image to HuggingFace's AI image detector model
    and return a verdict: FAKE, REAL, or SUSPICIOUS.
    """

    # Step 1: Check the image file exists
    print(f"[Step 1] Reading image from: {image_path}")
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return {"score": 0.5, "verdict": "SUSPICIOUS", "error": "File not found"}

    file_size = os.path.getsize(image_path)
    print(f"[Step 1] Image found ({file_size} bytes)")

    # Step 2: Check API key
    print(f"[Step 2] API key loaded: {'Yes' if HF_API_KEY else 'NO — key is missing!'}")
    if not HF_API_KEY:
        return {"score": 0.5, "verdict": "SUSPICIOUS", "error": "HF_API_KEY not set in .env"}

    # Step 3: Send to HuggingFace
    print(f"[Step 3] Sending image to HuggingFace model: {MODEL_NAME}...")
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Step 3] Attempt {attempt}/{max_retries}...")
            result = client.image_classification(image_path, model=MODEL_NAME)
            print(f"[Step 4] HuggingFace response: {result}")
            break
        except Exception as e:
            error_msg = str(e)
            print(f"[Step 3] Error: {error_msg[:200]}")

            # If model is loading, wait and retry
            if "503" in error_msg or "loading" in error_msg.lower():
                import time
                print(f"[Step 3] Model is loading... waiting 20 seconds")
                time.sleep(20)
                continue
            else:
                print(f"[ERROR] Detection failed: {error_msg[:300]}")
                return {"score": 0.5, "verdict": "SUSPICIOUS", "error": "Detection failed"}
    else:
        print("[ERROR] Model did not load after 3 attempts")
        return {"score": 0.5, "verdict": "SUSPICIOUS", "error": "Model loading timeout"}

    # Step 5: Find the "artificial" (fake) score
    fake_score = 0.5  # default
    for item in result:
        label = item.label.lower()
        if "artificial" in label or "fake" in label or "deepfake" in label:
            fake_score = round(item.score, 4)
            break
        elif "human" in label or "real" in label:
            # If we find the "real/human" score, Fake = 1 - Real
            fake_score = round(1.0 - item.score, 4)
            break
    print(f"[Step 5] Fake/AI score: {fake_score}")

    # Step 6: Determine the verdict
    if fake_score > 0.65:
        verdict = "FAKE"
    elif fake_score < 0.35:
        verdict = "REAL"
    else:
        verdict = "SUSPICIOUS"
    print(f"[Step 6] Verdict: {verdict}")

    # Step 7: Return the result
    final = {"score": fake_score, "verdict": verdict}
    print(f"[Step 7] Final result: {final}")
    return final


# --- Quick test (only runs if you execute this file directly) ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python detector.py <path_to_image>")
        print("Example: python detector.py test_photo.jpg")
    else:
        result = detect_deepfake(sys.argv[1])
        print("\n=== FINAL RESULT ===")
        print(result)
