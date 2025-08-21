import os
import re
import json
import logging
from functools import lru_cache
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightweightHybridDetector:
    """
    Lightweight detector that combines:
    - A simple keyword-based local heuristic (pure Python)
    - Gemini API (if available)
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LightweightHybridDetector, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        load_dotenv()
        self.gemini_model = None
        self.local_model = True  # flag for availability
        self._setup_gemini()
        self._initialized = True

    def _setup_gemini(self):
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")
                logger.info("✅ Gemini API initialized")
            else:
                logger.warning("⚠️ GEMINI_API_KEY not found")
        except Exception as e:
            logger.error(f"❌ Gemini setup failed: {str(e)}")

    @lru_cache(maxsize=500)
    def _predict_local(self, text: str) -> Tuple[str, float]:
        """Very small heuristic model using keyword rules."""
        t = text.lower()
        # Burmese + English spam/scam/phishing cues
        phishing_kw = [
            "verify", "login", "account", "password", "link", "recent login",
            "အကောင့်", "လင့်ခ်", "အတည်ပြု", "password", "ချက်ချင်း", "verify your",
        ]
        scam_kw = [
            "money", "transfer", "urgent", "help", "bank", "blocked",
            "ငွေ", "ချေး", "အကူအညီ", "အန္တရာယ်", "blocked", "hack",
        ]
        spam_kw = [
            "free", "win", "offer", "promo", "lottery", "congratulations",
            "အခမဲ့", "ဆုလာဘ်", "ကမ်းလှမ်းချက်", "ရရှိ", "သင်သည်", "iPhone",
        ]
        legit_kw = [
            "မင်္ဂလာပါ", "မနက်ဖန်", "ညနေ", "အစည်းအဝေး", "appointment",
            "coffee", "movie", "class", "meeting", "schedule",
        ]

        score = {"phishing": 0, "scam": 0, "spam": 0, "legitimate": 0}
        for kw in phishing_kw:
            if kw in t:
                score["phishing"] += 1
        for kw in scam_kw:
            if kw in t:
                score["scam"] += 1
        for kw in spam_kw:
            if kw in t:
                score["spam"] += 1
        for kw in legit_kw:
            if kw in t:
                score["legitimate"] += 1

        # Simple URL detection increases phishing/spam
        if re.search(r"https?://|www\\.", t):
            score["phishing"] += 1
            score["spam"] += 1

        # Choose top category
        category = max(score.items(), key=lambda x: x[1])[0]
        max_score = score[category]
        total_hits = sum(score.values())
        # Map to a soft confidence
        if total_hits == 0:
            return "legitimate", 0.55
        base = 0.6 + min(max_score, 3) * 0.1
        return category, min(base, 0.9)

    def _predict_gemini(self, text: str) -> Tuple[str, float, str]:
        try:
            if not self.gemini_model:
                return "unknown", 0.0, "Gemini API not available"
            prompt = f"""
            Analyze this Burmese text for spam detection. Respond in JSON format only:
            {{
              "category": "legitimate|spam|scam|phishing",
              "confidence": 0.0-1.0,
              "reasoning": "brief explanation"
            }}
            Text: "{text}"
            """
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=200
                )
            )
            try:
                response_text = response.text.strip()
                if '```json' in response_text:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    json_text = response_text[json_start:json_end]
                else:
                    json_text = response_text
                result = json.loads(json_text)
                category = result.get('category', 'unknown')
                confidence = float(result.get('confidence', 0.0))
                reasoning = result.get('reasoning', '')
                return category, confidence, reasoning
            except Exception:
                rl = response.text.lower()
                if 'phishing' in rl:
                    return 'phishing', 0.7, response.text[:120]
                if 'scam' in rl:
                    return 'scam', 0.7, response.text[:120]
                if 'spam' in rl:
                    return 'spam', 0.7, response.text[:120]
                return 'legitimate', 0.6, response.text[:120]
        except Exception as e:
            logger.error(f"❌ Gemini prediction failed: {str(e)}")
            return "unknown", 0.0, f"Error: {str(e)}"

    def _assess_risk(self, category: str, confidence: float) -> str:
        if category in ['scam', 'phishing']:
            if confidence > 0.8:
                return "very_high"
            elif confidence > 0.6:
                return "high"
            return "medium"
        if category == 'spam':
            return "medium" if confidence > 0.8 else "low"
        return "very_low"

    def predict_hybrid(self, text: str) -> Dict:
        local_cat, local_conf = self._predict_local(text)
        gem_cat, gem_conf, gem_reason = self._predict_gemini(text)

        local_weight = 0.6
        gem_weight = 0.4
        if local_cat == gem_cat and gem_cat != 'unknown':
            final_cat = local_cat
            final_conf = local_conf * local_weight + gem_conf * gem_weight
            agreement = "high"
        else:
            # prefer whichever has higher confidence; fall back to local
            if gem_conf >= local_conf:
                final_cat, final_conf = gem_cat, gem_conf * 0.85
            else:
                final_cat, final_conf = local_cat, local_conf * 0.85
            agreement = "low"

        risk = self._assess_risk(final_cat, final_conf)
        return {
            "final_prediction": {
                "category": final_cat,
                "confidence": round(final_conf, 3),
                "risk_level": risk,
                "agreement": agreement,
            },
            "local_model": {
                "category": local_cat,
                "confidence": round(local_conf, 3)
            },
            "gemini_api": {
                "category": gem_cat,
                "confidence": round(gem_conf, 3),
                "reasoning": gem_reason
            },
            "input_text": text,
            "model_version": "lightweight_hybrid_v1"
        }

    def get_model_stats(self) -> Dict:
        return {
            "categories": ["legitimate", "spam", "scam", "phishing"],
            "local_model_available": True,
            "gemini_api_available": self.gemini_model is not None,
            "deployment": "serverless_light"
        }

_detector_instance = None

def get_detector():
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = LightweightHybridDetector()
    return _detector_instance
