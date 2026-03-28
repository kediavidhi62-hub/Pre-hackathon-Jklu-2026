import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# System prompt for the AI forensics expert
SYSTEM_PROMPT = """You are a senior digital forensics analyst with 15 years 
of experience detecting AI-generated and manipulated media.

Rules:
- Write exactly 3 sentences
- Sentence 1: State what the score means technically.
  Mention GAN artifacts, facial boundary inconsistencies,
  or skin texture anomalies based on the score level
- Sentence 2: Explain what specifically causes scores 
  in this range. Mention the actual score number
- Sentence 3: Give a clear actionable recommendation
- Never use bullet points
- Never say I cannot see the image
- Sound authoritative and specific
- Output only the 3 sentences nothing else"""

# Fallback explanations if Gemini fails
FALLBACK = {
    "FAKE": "High manipulation indicators detected. Exercise caution.",
    "REAL": "No significant manipulation detected. Content appears authentic.",
    "SUSPICIOUS": "Inconclusive results. Seek additional verification.",
}


def generate_explanation(score: float, verdict: str, file_type: str = "image") -> str:
    """
    Use Gemini to generate a beginner-friendly forensic explanation
    of the deepfake detection result.
    """

    # Step 1: Build the user message
    user_message = f"""A deepfake detection model analyzed a {file_type} and 
returned a fake probability score of {score:.2f} out of 
1.0 with verdict: {verdict}.

Score context:
- Above 0.75 = strongly indicates AI manipulation
- Between 0.25 and 0.75 = uncertain or partial manipulation
- Below 0.25 = strongly indicates authentic media

Write a 3 sentence forensic report for score {score:.2f}.
Be specific. Do not invent pixel level details."""
    print(f"[Explainer] Asking Gemini to explain: score={score}, verdict={verdict}")
    print(f"[Explainer] User message: {user_message}")

    # Step 2: Check API key
    if not GEMINI_API_KEY:
        print("[Explainer] ERROR: GEMINI_API_KEY not set in .env!")
        return FALLBACK.get(verdict, FALLBACK["SUSPICIOUS"])

    # Step 3: Call Gemini
    try:
        print("[Explainer] Calling Gemini API...")
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=SYSTEM_PROMPT,
        )

        response = model.generate_content(
            user_message,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=200,
            ),
        )

        explanation = response.text.strip()
        print(f"[Explainer] Gemini response: {explanation}")
        return explanation

    except Exception as e:
        print(f"[Explainer] ERROR: Gemini failed — {e}")
        fallback_text = FALLBACK.get(verdict, FALLBACK["SUSPICIOUS"])
        print(f"[Explainer] Using fallback: {fallback_text}")
        return fallback_text


# --- Quick test (only runs if you execute this file directly) ---
if __name__ == "__main__":
    print("=== Testing Explainer ===\n")

    # Test 1: FAKE verdict
    print("--- Test 1: FAKE ---")
    result1 = generate_explanation(0.91, "FAKE", "image")
    print(f"Result: {result1}\n")

    # Test 2: REAL verdict
    print("--- Test 2: REAL ---")
    result2 = generate_explanation(0.04, "REAL", "image")
    print(f"Result: {result2}\n")

    # Test 3: SUSPICIOUS verdict
    print("--- Test 3: SUSPICIOUS ---")
    result3 = generate_explanation(0.50, "SUSPICIOUS", "image")
    print(f"Result: {result3}\n")
