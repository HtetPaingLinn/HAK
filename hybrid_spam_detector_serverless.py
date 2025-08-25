import os
import json
import logging
import re
from typing import Tuple, Dict, Any, List, Optional

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
        self.kb: Dict[str, Any] = {}
        # Holds last structured Gemini JSON (if available)
        self._last_gemini_context: Optional[Dict[str, Any]] = None

        # Load KB JSON (rule-based indicators)
        try:
            kb_path = os.path.join(os.path.dirname(__file__), "kb_burmese_rules.json")
            if os.path.exists(kb_path):
                with open(kb_path, "r", encoding="utf-8") as f:
                    self.kb = json.load(f)
                logger.info("Loaded Burmese KB rules for spam detection")
            else:
                logger.warning("KB file kb_burmese_rules.json not found; rule-based features disabled")
        except Exception as e:
            logger.error(f"Failed to load KB rules: {e}")
            self.kb = {}

        # Initialize Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY is not set. Gemini-based predictions will not work.")
        else:
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
        Hybrid prediction combining a lightweight rule-based KB with Gemini.
        - Rule-based provides deterministic cues and base confidence
        - Gemini provides semantic reasoning and category
        """
        # Rule-based prediction (does not require external API)
        rule_out = self._predict_rules(text)

        # Gemini prediction (optional)
        gem_category: Optional[str] = None
        gem_conf: Optional[float] = None
        gem_reason: Optional[str] = None
        if self.gemini_model is not None:
            try:
                gem_category, gem_conf, gem_reason = self._predict_gemini(text)
            except Exception as e:
                logger.error(f"Gemini error in hybrid path: {e}")

        # Decide final category
        final_category = rule_out["category"]
        final_confidence = rule_out["confidence"]
        agreement = None

        if gem_category is not None and gem_conf is not None:
            agreement = (final_category == gem_category)
            # If categories agree, boost confidence
            if agreement:
                final_confidence = min(1.0, (rule_out["confidence"] * 0.5 + gem_conf * 0.6))
                final_category = gem_category
            else:
                # Disagree: choose higher confidence source
                if gem_conf >= rule_out["confidence"]:
                    final_category = gem_category
                    final_confidence = gem_conf * 0.95
                else:
                    final_category = rule_out["category"]
                    final_confidence = rule_out["confidence"] * 0.95

        # Compute spam probability (0..1) blending KB scores and Gemini
        spam_like = {"Spam", "Scam", "Phishing"}
        # From KB scores (max of spam-like categories)
        kb_scores = rule_out.get("scores") or {"Spam": 0.0, "Scam": 0.0, "Phishing": 0.0, "Legit": 0.0}
        spam_prob_kb = float(max(kb_scores.get("Spam", 0.0), kb_scores.get("Scam", 0.0), kb_scores.get("Phishing", 0.0)))
        # From Gemini category/confidence if available
        if gem_category is not None and gem_conf is not None:
            spam_prob_gem = float(gem_conf if gem_category in spam_like else (1.0 - gem_conf))
            # Blend with slight emphasis on Gemini for semantics
            spam_probability = max(0.0, min(1.0, 0.6 * spam_prob_gem + 0.4 * spam_prob_kb))
        else:
            spam_probability = max(0.0, min(1.0, spam_prob_kb))

        binary_label = "spam" if spam_probability >= 0.5 else "general"

        details: Dict[str, Any] = {
            "kb_hits": rule_out.get("hits", []),
            "kb_score": rule_out.get("confidence", 0.0),
        }
        if gem_reason is not None:
            details["reasoning"] = gem_reason
        if agreement is not None:
            details["agreement"] = agreement
        # Attach structured Gemini context if available
        if getattr(self, "_last_gemini_context", None):
            details["gemini_structured"] = self._last_gemini_context

        return {
            "final_prediction": {
                "label": final_category,
                "confidence": float(final_confidence),
                "source": "hybrid_kb_gemini" if gem_category is not None else "kb_only"
            },
            # Binary view for quick UI decisions
            "binary_label": binary_label,  # "general" or "spam"
            "spam_probability": float(spam_probability),  # 0..1
            "spam_probability_pct": round(float(spam_probability) * 100, 2),  # 0..100
            "rule_based": rule_out,
            "gemini_api": (
                {
                    "category": gem_category,
                    "confidence": gem_conf,
                    "reasoning": gem_reason,
                    "structured": getattr(self, "_last_gemini_context", None),
                }
                if gem_category is not None else None
            ),
            "details": details,
        }

    def _predict_gemini(self, text: str) -> Tuple[str, float, str]:
        if not self.gemini_model:
            raise RuntimeError("Gemini model is not available. Ensure GEMINI_API_KEY is set.")
        # Ask for structured Burmese JSON with richer taxonomy
        prompt = f"""
        အောက်ပါ မြန်မာစာသားကို လုံခြုံရေး ရှုထောင့်မှ ခွဲခြားပါ။ မြန်မာဘာသာဖြင့် JSON အဖြစ်သာ ထုတ်ပေးပါ။

        အဓိကအမျိုးအစားများ (primary_category):
        - Legit (ယုံကြည်စိတ်ချရ)
        - Spam (စပမ်)
        - Scam (လှည်ဖြားမှု)
        - Phishing (ဖစ်ရှင်း)
        - Malware (မေါ်လဝဲ)
        - Impersonation (မည်သူ့အဖြစ် ဖျက်ဆီးယှက်ယှဲ)
        - Misinformation (မှားယွင်းသတင်း)
        - Harassment (အနှောင့်အယှက်/သာမန်မဟုတ်သောဖော်ပြချက်)

        Subcategory ဥပမာများ (optional):
        - Scam: investment, lottery, job, romance, tech_support, delivery, loan
        - Phishing: credential, otp, payment, account_verification, bank, wallet
        - Spam: marketing, bulk, referral, affiliate
        - Malware: apk, link, attachment, drive_by
        - Impersonation: official, bank, company, celebrity, friend_family

        JSON Schema:
        {
          "primary_category": "Legit|Spam|Scam|Phishing|Malware|Impersonation|Misinformation|Harassment",
          "subcategory": "string",
          "confidence": 0.0-1.0,
          "risk_level": "Low|Medium|High|Critical",
          "reasons_mm": ["မြန်မာဘာသာဖြင့် အကြောင်းပြချက်များ"],
          "red_flags_mm": ["သံသယဖြစ်စရာအချက်များ"],
          "suggested_actions_mm": ["လုံခြုံရေးအတွက် လုပ်ဆောင်ရန်"],
          "short_explanation_mm": "အကျဉ်းချုံးရှင်းလင်းချက်"
        }

        စာသား:
        '''{text}'''
        """
        try:
            response = self.gemini_model.generate_content(prompt)
            raw = (response.text or "").strip()
        except Exception as e:
            logger.error(f"Gemini prediction failed: {e}")
            raise
        # Try to parse structured JSON from model output
        self._last_gemini_context = None
        parsed_category: Optional[str] = None
        parsed_conf: Optional[float] = None
        # Extract JSON if wrapped in code fences
        candidate = raw
        if "```" in raw:
            try:
                candidate = raw.split("```", 2)[1]
                # Remove language tag if present
                candidate = candidate.split("\n", 1)[1] if candidate.startswith("json") else candidate
            except Exception:
                candidate = raw
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                self._last_gemini_context = obj
                pc = str(obj.get("primary_category") or "").strip()
                # Normalize category capitalization
                mapping = {
                    "legit": "Legit",
                    "spam": "Spam",
                    "scam": "Scam",
                    "phishing": "Phishing",
                    "malware": "Malware",
                    "impersonation": "Impersonation",
                    "misinformation": "Misinformation",
                    "harassment": "Harassment",
                }
                parsed_category = mapping.get(pc.lower(), "Legit") if pc else None
                conf = obj.get("confidence")
                if isinstance(conf, (int, float)):
                    parsed_conf = max(0.0, min(1.0, float(conf)))
        except Exception:
            # Fallback to heuristic if JSON parsing fails
            parsed_category = None
            parsed_conf = None

        if parsed_category is None or parsed_conf is None:
            # Heuristic extraction
            lower = raw.lower()
            if "phishing" in lower or "ဖစ်ရှင်း" in lower:
                parsed_category = "Phishing"
            elif "scam" in lower or "လှည်ဖြား" in lower:
                parsed_category = "Scam"
            elif "malware" in lower or "မေါ်လဝဲ" in lower:
                parsed_category = "Malware"
            elif "impersonation" in lower or "အဖြစ်" in lower:
                parsed_category = "Impersonation"
            elif "spam" in lower or "စပမ်" in lower:
                parsed_category = "Spam"
            else:
                parsed_category = "Legit"
            parsed_conf = 0.75 if parsed_category != "Legit" else 0.7

        return parsed_category, float(parsed_conf), raw

    # -------------------- Rule-based (KB) prediction --------------------
    def _predict_rules(self, text: str) -> Dict[str, Any]:
        """Lightweight pattern scoring using KB. Returns category, confidence, and hit details.
        Confidence in [0,1] based on matched keyword weights and strong phrases.
        """
        if not self.kb:
            # No KB -> neutral Lean Legit
            # Provide default scores for downstream spam probability calc
            return {"category": "Legit", "confidence": 0.5, "hits": [], "scores": {"Phishing": 0.0, "Scam": 0.0, "Spam": 0.0, "Legit": 0.5}}

        lower = text.lower()
        hits: List[Dict[str, Any]] = []

        # Strong phrases override
        strong = self.kb.get("strong_phrases", {})
        for cat, phrases in strong.items():
            for p in phrases:
                if p.lower() in lower:
                    hits.append({"category": cat, "phrase": p, "strength": "strong"})
        if hits:
            # If any strong phrase hits, choose the most frequent category among them
            cat_counts: Dict[str, int] = {}
            for h in hits:
                cat_counts[h["category"]] = cat_counts.get(h["category"], 0) + 1
            best_cat = max(cat_counts.items(), key=lambda x: x[1])[0]
            # Provide scores reflecting a strong override
            scores_override: Dict[str, float] = {"Phishing": 0.0, "Scam": 0.0, "Spam": 0.0, "Legit": 0.0}
            scores_override[best_cat] = 0.9
            return {"category": best_cat, "confidence": 0.9, "hits": hits, "scores": scores_override}

        # Keyword-based scoring
        scores: Dict[str, float] = {"Phishing": 0.0, "Scam": 0.0, "Spam": 0.0, "Legit": 0.0}
        cats = self.kb.get("categories", {})
        for cat, cfg in cats.items():
            kw: List[str] = cfg.get("keywords", [])
            weight: float = float(cfg.get("weight", 0.6))
            for k in kw:
                if k.strip() and k.lower() in lower:
                    scores[cat] = min(1.0, scores.get(cat, 0.0) + weight * 0.5)
                    hits.append({"category": cat, "keyword": k, "strength": "keyword"})

        # Ham indicators slightly push towards Legit
        for ham in self.kb.get("ham_indicators", []):
            if ham.lower() in lower:
                scores["Legit"] = min(1.0, scores["Legit"] + 0.15)
                hits.append({"category": "Legit", "keyword": ham, "strength": "ham"})

        # Choose best category
        best_cat = max(scores.items(), key=lambda x: x[1])[0]
        best_score = scores[best_cat]
        # Apply base floors
        if best_cat == "Legit" and best_score < 0.55:
            best_score = 0.55
        elif best_cat != "Legit" and best_score < 0.65:
            best_score = 0.65

        return {"category": best_cat, "confidence": float(best_score), "hits": hits, "scores": scores}

    def get_model_stats(self) -> Dict[str, Any]:
        return {
            "serverless": True,
            "local_model_available": self.local_model is not None,
            "gemini_model_available": self.gemini_model is not None,
            "mode": "gemini_only",
        }


def get_detector() -> ServerlessHybridDetector:
    return ServerlessHybridDetector()
