import os
import requests
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

if not HF_API_KEY:
    print("WARNING: HF_API_KEY not found in environment variables. Model queries will likely fail unauthorized.")

# HuggingFace Model endpoints
MODELS = {
    "umm-maybe": "https://api-inference.huggingface.co/models/umm-maybe/AI-image-detector",
    "dima806": "https://api-inference.huggingface.co/models/dima806/deepfake_vs_real_image_detection"
}

HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}

def query_model(model_url, image_bytes, model_name, image_desc):
    """Queries a specific HuggingFace model with image bytes."""
    print(f"[{model_name}] Sending {image_desc} to API...")
    try:
        response = requests.post(model_url, headers=HEADERS, data=image_bytes)
        if response.status_code == 200:
            result = response.json()
            print(f"[{model_name}] {image_desc} response: {result}")
            return parse_score(result, model_name)
        else:
            print(f"[{model_name}] Error {response.status_code} on {image_desc}: {response.text}")
            return None
    except Exception as e:
        print(f"[{model_name}] Exception on {image_desc}: {e}")
        return None

def parse_score(result, model_name):
    """Parses the API response to extract the probability of the image being FAKE (0.0 to 1.0)."""
    try:
        # HuggingFace responses can sometimes be wrapped in a nested list
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
            result = result[0]
            
        if isinstance(result, list):
            fake_score = 0.5 # Default fallback
            for item in result:
                label = str(item.get("label", "")).lower()
                score = float(item.get("score", 0.0))
                
                # Identify if the label means 'fake' or 'real'
                if "fake" in label or "artificial" in label:
                    return score # Found direct fake score
                elif "real" in label or "human" in label:
                    fake_score = 1.0 - score # Inferred fake score from real probability
            return fake_score
    except Exception as e:
        print(f"[{model_name}] Error parsing score from result {result}: {e}")
    
    return None

def process_image(image_path):
    """Processes an image, scoring the original and a 60% center crop across two models."""
    print(f"\n--- Processing Image: {image_path} ---")
    
    try:
        with Image.open(image_path) as img:
            # Ensure consistent RGB format
            img = img.convert("RGB")
            
            # --- 1. Get bytes for the original image ---
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='JPEG')
            original_bytes = img_byte_arr.getvalue()
            
            # --- 2. Create the 60% Center Crop ---
            width, height = img.size
            crop_width = int(width * 0.6)
            crop_height = int(height * 0.6)
            
            left = (width - crop_width) // 2
            top = (height - crop_height) // 2
            right = left + crop_width
            bottom = top + crop_height
            
            print(f"Original size: {width}x{height}. Cropping to 60%: {crop_width}x{crop_height} at ({left}, {top})")
            img_cropped = img.crop((left, top, right, bottom))
            
            crop_byte_arr = BytesIO()
            img_cropped.save(crop_byte_arr, format='JPEG')
            cropped_bytes = crop_byte_arr.getvalue()
            
    except Exception as e:
        print(f"Failed to load or process image {image_path}: {e}")
        return {"score": 0.5, "verdict": "SUSPICIOUS"}

    scores = []
    
    # --- 3. Run all models on both the original and cropped versions ---
    for model_name, url in MODELS.items():
        # Query original
        score_orig = query_model(url, original_bytes, model_name, "original image")
        if score_orig is not None:
            scores.append(score_orig)
            
        # Query cropped
        score_crop = query_model(url, cropped_bytes, model_name, "cropped image")
        if score_crop is not None:
            scores.append(score_crop)
            
    print(f"\nCollected successful fake scores: {scores}")
    
    # --- 4. Fallback if everything fails ---
    if len(scores) == 0:
        print("All models failed to return a valid score. Defaulting to SUSPICIOUS.")
        return {"score": 0.5, "verdict": "SUSPICIOUS"}
        
    # --- 5. Calculate Final Score ---
    final_score = sum(scores) / len(scores)
    print(f"Final Average Fake Score: {final_score:.4f}")
    
    # --- 6. Verdict Thresholds ---
    if final_score > 0.75:
        verdict = "FAKE"
    elif final_score < 0.25:
        verdict = "REAL"
    else:
        verdict = "SUSPICIOUS"
        
    print(f"Final Verdict: {verdict}")
    print("-------------------------------------------\n")
    
    return {
        "score": final_score,
        "verdict": verdict
    }

if __name__ == "__main__":
    # Simple CLI usage for debugging
    import sys
    if len(sys.argv) > 1:
        process_image(sys.argv[1])
    else:
        print("Usage: python detector.py <path_to_image>")