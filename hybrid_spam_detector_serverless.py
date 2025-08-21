import os
import logging
from typing import Tuple, Dict, Any

import google.generativeai as genai

logger = logging.getLogger(__name__)


class ServerlessHybridDetector:
    """
    Serverless-friendly detector that currently uses Gemini only.
    Provides the API surface expected by main.py:
      - predict_hybrid(text) -> dict with {"final_prediction": {"confidence": float}}
      - _predict_gemini(text) -> (category, confidence, reasoning)
      - get_model_stats() -> dict
      - attributes: local_model (None in serverless), gemini_model (truthy if configured)
    """

    def __init__(self) -> None:
        self.local_model = None  # Not available in serverless
        self.gemini_model = None

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY is not set. Gemini-based predictions will not work.")
            return

        try:
            genai.configure(api_key=api_key)
            # Use the same model name as used elsewhere in repo
            self.gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")
            logger.info("Gemini model initialized for serverless detector")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            self.gemini_model = None

    def predict_hybrid(self, text: str) -> Dict[str, Any]:
        """
        Placeholder hybrid method. Currently proxies to Gemini-only prediction
        and wraps the output in the expected structure.
        """
        category, confidence, reasoning = self._predict_gemini(text)
        # Map to an example schema the frontend/backend expects
        result = {
            "final_prediction": {
                "label": category,
                "confidence": float(confidence),
                "source": "gemini_only"
            },
            "details": {
                "reasoning": reasoning
            }
        }
        return result

    def _predict_gemini(self, text: str) -> Tuple[str, float, str]:
        if not self.gemini_model:
            raise RuntimeError("Gemini model is not available. Ensure GEMINI_API_KEY is set.")

        prompt = f"""
        မြန်မာစာပိုဒ်ကို စစ်ဆေးပါ။ အောက်ပါအမျိုးအစားများထဲမှ တစ်ခုအဖြစ် ခွဲခြားပါ:
        - ယုံကြည်စိတ်ချရ (Legit)
        - စပမ် (Spam)
        - လှည်ဖြားမှု (Scam)
        - ဖစ်ရှင်း (Phishing)

        အမျိုးအစားနှင့် အကြောင်းပြချက်ကို မြန်မာဘာသာဖြင့် ပေးပါ။

        စာသား:
        '''{text}'''
        """
        try:
            response = self.gemini_model.generate_content(prompt)
            raw = (response.text or "").strip()
        except Exception as e:
            logger.error(f"Gemini prediction failed: {e}")
            raise

        # Very simple heuristic extraction from model output
        lower = raw.lower()
        if "phishing" in lower or "ဖစ်ရှင်း" in lower:
            category = "Phishing"
        elif "scam" in lower or "လှည်ဖြား" in lower:
            category = "Scam"
        elif "spam" in lower or "စပမ်" in lower:
            category = "Spam"
        else:
            category = "Legit"

        # We don't have a numeric score; assign a heuristic confidence
        if category == "Legit":
            confidence = 0.7
        else:
            confidence = 0.75

        return category, confidence, raw

    def get_model_stats(self) -> Dict[str, Any]:
        return {
            "serverless": True,
            "local_model_available": self.local_model is not None,
            "gemini_model_available": self.gemini_model is not None,
            "mode": "gemini_only",
        }


def get_detector() -> ServerlessHybridDetector:
    return ServerlessHybridDetector()
