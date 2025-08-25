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
        gem_detailed: Optional[Dict[str, Any]] = None
        if self.gemini_model is not None:
            try:
                # Prefer detailed prediction (JSON). Fallback to simple.
                try:
                    gem_detailed = self.predict_gemini_detailed(text)
                    gem_category = gem_detailed.get("primary_label") or gem_detailed.get("category")
                    gem_conf = float(gem_detailed.get("confidence", gem_detailed.get("spam_probability", 0.75)))
                    gem_reason = gem_detailed.get("rationale") or gem_detailed.get("reasoning")
                except Exception:
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
                    **({} if gem_detailed is None else {
                        "primary_label": gem_detailed.get("primary_label"),
                        "labels": gem_detailed.get("labels"),
                        "spam_probability": gem_detailed.get("spam_probability"),
                        "binary_label": gem_detailed.get("binary_label"),
                        "rationale": gem_detailed.get("rationale"),
                        "risk_factors": gem_detailed.get("risk_factors"),
                        "suggested_actions": gem_detailed.get("suggested_actions"),
                        "evidence_phrases": gem_detailed.get("evidence_phrases"),
                        "urls": gem_detailed.get("urls"),
                        "has_links": gem_detailed.get("has_links"),
                        "contains_personal_info": gem_detailed.get("contains_personal_info"),
                        "language": gem_detailed.get("language"),
                        "severity": gem_detailed.get("severity"),
                    })
                }
                if gem_category is not None else None
            ),
            "details": details,
        }

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

    def predict_gemini_detailed(self, text: str) -> Dict[str, Any]:
        """Ask Gemini to return a structured JSON with rich context and multi-label confidences."""
        if not self.gemini_model:
            raise RuntimeError("Gemini model is not available. Ensure GEMINI_API_KEY is set.")

        schema_hint = (
            "Return ONLY JSON with keys: primary_label (one of: Legit, Spam, Scam, Phishing, "
            "Promotion, Malware, Adult, Impersonation, Investment, Lottery, TechSupport, Loan, Crypto), "
            "labels (object of label->confidence 0..1), spam_probability (0..1), binary_label ('general'|'spam'), "
            "rationale (string), risk_factors (array of strings), suggested_actions (array of strings), "
            "evidence_phrases (array of strings), urls (array of strings), has_links (bool), "
            "contains_personal_info (bool), language (string), severity (1..5)."
        )

        prompt = f"""
        မြန်မာစာပိုဒ်ကို စစ်ဆေးပြီး အောက်ပါ format နည်းလမ်းအတိုင်း JSON တစ်ခုတည်းသာ ထုတ်ပေးပါ။
        {schema_hint}

        စာသား:
        '''{text}'''
        """
        try:
            response = self.gemini_model.generate_content(prompt)
            raw = (response.text or "").strip()
        except Exception as e:
            logger.error(f"Gemini detailed prediction failed: {e}")
            raise

        # Try parse JSON strictly
        parsed: Dict[str, Any]
        try:
            # Extract JSON block if model added extra text
            start = raw.find('{')
            end = raw.rfind('}')
            json_str = raw[start:end+1] if start != -1 and end != -1 else raw
            parsed = json.loads(json_str)
        except Exception:
            # Fallback: derive from simple classifier
            cat, conf, reason = self._predict_gemini(text)
            parsed = {
                "primary_label": cat,
                "labels": {cat: conf},
                "spam_probability": conf if cat in {"Spam", "Scam", "Phishing"} else 1.0 - conf,
                "binary_label": "spam" if cat in {"Spam", "Scam", "Phishing"} else "general",
                "rationale": reason,
                "risk_factors": [],
                "suggested_actions": [],
                "evidence_phrases": [],
                "urls": [],
                "has_links": False,
                "contains_personal_info": False,
                "language": "my",
                "severity": 3,
            }

        # Normalize some fields and compute a single confidence representative
        primary = parsed.get("primary_label")
        labels = parsed.get("labels") or {}
        if primary and primary in labels:
            parsed["confidence"] = float(labels.get(primary, 0.75))
        else:
            parsed["confidence"] = float(parsed.get("spam_probability", 0.75))

        return parsed

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
