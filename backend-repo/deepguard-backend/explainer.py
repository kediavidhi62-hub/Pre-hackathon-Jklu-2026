import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# System prompt for the AI forensics expert
SYSTEM_PROMPT = (
    "You are a digital forensics AI assistant specializing in deepfake "
    "detection. Generate clear forensic reports for non-technical users. "
    "Rules: plain English, authoritative but not alarmist, exactly 3 "
    "sentences, no bullet points, no titles, output only the paragraph. "
    "Never invent specific pixel details. Base analysis on score ranges only."
)

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
    user_message = (
        f"A deepfake model analyzed a {file_type} and returned a fake "
        f"probability of {score * 100:.0f}%. Verdict: {verdict}. "
        f"Generate a forensic explanation based on this score range only."
    )
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
                max_output_tokens=1024,
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
