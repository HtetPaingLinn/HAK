import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
import joblib
import google.generativeai as genai
import os
from dotenv import load_dotenv
import re
import logging
from typing import Dict, Tuple, List
import json
import io
from functools import lru_cache
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServerlessHybridSpamDetector:
    """
    Serverless-optimized version of the hybrid spam detector
    Designed for Vercel deployment with cold start optimization
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServerlessHybridSpamDetector, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        load_dotenv()
        self.vectorizer = None
        self.local_model = None
        self.gemini_model = None
        # Control whether to use Gemini via env flag (default: False)
        self.use_gemini = os.getenv("USE_GEMINI", "false").lower() == "true"
        self.label_mapping = {
            'legitimate': 0,
            'spam': 1, 
            'scam': 2,
            'phishing': 3
        }
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        
        # Initialize components
        self._setup_gemini()
        self._initialize_local_model()
        
        self._initialized = True
    
    def _setup_gemini(self):
        """Setup Gemini API with error handling"""
        # Only initialize Gemini if explicitly enabled
        if not self.use_gemini:
            logger.info("ℹ️ USE_GEMINI is disabled. Running local-only model.")
            return
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
    
    def _get_default_dataset(self) -> str:
        """Return default dataset as CSV string for serverless deployment"""
        return """text,label,category,confidence
"သင့်အတွက် အထူးကမ်းလှမ်းချက်! ယခုပင် ဖုန်းနံပါတ်ကို ပေးပို့ပါ။",spam,promotional,0.95
"ဘဏ်အကောင့်ကို အတည်ပြုရန် လိုအပ်ပါသည်။ လင့်ခ်ကို နှိပ်ပါ။",phishing,financial,0.98
"မင်္ဂလာပါ၊ ဒီနေ့ အစည်းအဝေးကို ၃နာရီမှာ ရွှေ့ထားပါတယ်။",legitimate,business,0.92
"အခမဲ့ ဆုလာဘ်ရရှိရန် ယခုပင် စာရင်းသွင်းပါ!",spam,promotional,0.94
"သင့်ဖုန်းကို ဗိုင်းရပ်စ်ကူးစက်ထားပါသည်။ ချက်ချင်းဒေါင်းလုဒ်လုပ်ပါ။",scam,security,0.97
"မနက်ဖန် ရုံးမှာ တွေ့ကြမယ်နော်။ စာရွက်စာတမ်းတွေ မမေ့နဲ့။",legitimate,personal,0.89
"လူကြီးမင်း၊ ငွေကြေးအကူအညီ လိုအပ်ပါသည်။ ကျေးဇူးပြု၍ ကူညီပေးပါ။",scam,financial,0.93
"ဒီနေ့ ညနေ ၆နာရီမှာ အိမ်ပြန်မယ်။ ညစာ ပြင်ထားပါ။",legitimate,personal,0.91
"သင်သည် ၁သိန်းကျပ် အနိုင်ရရှိပါပြီ! ချက်ချင်း ဆက်သွယ်ပါ။",spam,lottery,0.96
"ဘဏ်မှ အရေးကြီးသတင်းကြား။ အကောင့်ကို ပိတ်ထားပါမည်။",phishing,financial,0.95
"ကျောင်းသားများအတွက် ပညာသင်ဆုများ ရရှိနိုင်ပါသည်။",legitimate,education,0.88
"သင့်အကောင့်ကို hack လုပ်ထားပါသည်။ password ပြောင်းပါ။",scam,security,0.94
"မနက်ဖန် ဆေးရုံတွင် ချိန်းဆိုထားပါသည်။ ၁၀နာရီ။",legitimate,medical,0.90
"အခမဲ့ iPhone ရရှိရန် လင့်ခ်ကို နှိပ်ပါ!",spam,promotional,0.97
"သင့်မိသားစုအတွက် အန္တရာယ်ရှိပါသည်။ ချက်ချင်း ဆက်သွယ်ပါ။",scam,threat,0.98
"ဒီနေ့ အလုပ်ပြီးရင် ကော်ဖီသောက်ကြမလား။",legitimate,personal,0.87
"သင့်ဖုန်းနံပါတ်သည် ဆုလာဘ်အနိုင်ရရှိပါပြီ!",spam,lottery,0.95
"ကုမ္ပဏီမှ အရေးကြီးကြေညာချက်။ ဝန်ထမ်းများ သိရှိရန်။",legitimate,business,0.86
"သင့်အား ငွေကြေးချေးငှားပေးနိုင်ပါသည်။ အတိုးနှုန်း ၅%။",spam,financial,0.92
"ရဲစခန်းမှ သတင်းကြား။ သင့်အမည်ဖြင့် အမှုရှိပါသည်။",scam,legal,0.96"""
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess Burmese text"""
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\u1000-\u109F\u1040-\u1049\s\w]', ' ', text)
        return text.lower()
    
    def _initialize_local_model(self):
        """Initialize local model from CSV dataset (falls back to embedded sample)"""
        try:
            # Try to load dataset from file first
            dataset_path = Path(__file__).parent / "burmese_spam_dataset.csv"
            if dataset_path.exists():
                df = pd.read_csv(dataset_path)
                logger.info(f"✅ Loaded dataset from {dataset_path} with {len(df)} rows")
            else:
                # Fallback to embedded minimal dataset
                csv_data = io.StringIO(self._get_default_dataset())
                df = pd.read_csv(csv_data)
                logger.warning("⚠️ Dataset file not found. Using embedded sample dataset.")
            
            # Preprocess texts
            df['processed_text'] = df['text'].apply(self._preprocess_text)
            df['label_num'] = df['label'].map(self.label_mapping)
            
            # Prepare features and labels
            X = df['processed_text'].values
            y = df['label_num'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Create TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=3000,  # allow richer features with larger dataset
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            
            # Fit vectorizer and transform data
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            
            # Create lightweight ensemble model
            nb_model = MultinomialNB(alpha=0.1)
            lr_model = LogisticRegression(random_state=42, max_iter=500)
            
            self.local_model = VotingClassifier(
                estimators=[
                    ('naive_bayes', nb_model),
                    ('logistic_regression', lr_model)
                ],
                voting='soft'
            )
            
            # Train model
            self.local_model.fit(X_train_tfidf, y_train)
            
            logger.info("✅ Local model initialized for serverless")
            
        except Exception as e:
            logger.error(f"❌ Local model initialization failed: {str(e)}")
    
    @lru_cache(maxsize=100)
    def _predict_local_cached(self, text: str) -> Tuple[str, float]:
        """Cached local prediction for better performance"""
        return self._predict_local(text)
    
    def _predict_local(self, text: str) -> Tuple[str, float]:
        """Predict using local ML model"""
        try:
            if not self.local_model or not self.vectorizer:
                return "unknown", 0.0
            
            processed_text = self._preprocess_text(text)
            text_tfidf = self.vectorizer.transform([processed_text])
            
            prediction = self.local_model.predict(text_tfidf)[0]
            probabilities = self.local_model.predict_proba(text_tfidf)[0]
            confidence = max(probabilities)
            
            label = self.reverse_label_mapping.get(prediction, "unknown")
            return label, confidence
            
        except Exception as e:
            logger.error(f"❌ Local prediction failed: {str(e)}")
            return "unknown", 0.0
    
    def _predict_gemini(self, text: str) -> Tuple[str, float, str]:
        """Predict using Gemini API with timeout handling"""
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
            
            # Parse JSON response
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
                
            except (json.JSONDecodeError, ValueError):
                # Fallback parsing
                response_lower = response.text.lower()
                if 'spam' in response_lower:
                    return 'spam', 0.7, response.text[:100]
                elif 'scam' in response_lower:
                    return 'scam', 0.7, response.text[:100]
                elif 'phishing' in response_lower:
                    return 'phishing', 0.7, response.text[:100]
                else:
                    return 'legitimate', 0.6, response.text[:100]
            
        except Exception as e:
            logger.error(f"❌ Gemini prediction failed: {str(e)}")
            return "unknown", 0.0, f"Error: {str(e)}"
    
    def predict_hybrid(self, text: str) -> Dict:
        """
        Serverless-optimized hybrid prediction
        """
        # Get local prediction (cached)
        local_category, local_confidence = self._predict_local_cached(text)

        # If Gemini is disabled or unavailable, use local-only prediction
        if not self.use_gemini or not self.gemini_model:
            risk_level = self._assess_risk(local_category, local_confidence)
            return {
                "final_prediction": {
                    "category": local_category,
                    "confidence": round(local_confidence, 3),
                    "risk_level": risk_level,
                    "agreement": "n/a"
                },
                "local_model": {
                    "category": local_category,
                    "confidence": round(local_confidence, 3)
                },
                "gemini_api": {
                    "category": "unknown",
                    "confidence": 0.0,
                    "reasoning": "Gemini disabled"
                },
                "input_text": text,
                "model_version": "local_only_v1.1"
            }

        # Otherwise, combine with Gemini
        gemini_category, gemini_confidence, gemini_reasoning = self._predict_gemini(text)
        local_weight = 0.6
        gemini_weight = 0.4

        if local_category == gemini_category:
            final_category = local_category
            final_confidence = (local_confidence * local_weight + gemini_confidence * gemini_weight)
            agreement = "high"
        else:
            if local_confidence > gemini_confidence:
                final_category = local_category
                final_confidence = local_confidence * 0.8
            else:
                final_category = gemini_category
                final_confidence = gemini_confidence * 0.8
            agreement = "low"
        
        # Risk assessment
        risk_level = self._assess_risk(final_category, final_confidence)
        
        return {
            "final_prediction": {
                "category": final_category,
                "confidence": round(final_confidence, 3),
                "risk_level": risk_level,
                "agreement": agreement
            },
            "local_model": {
                "category": local_category,
                "confidence": round(local_confidence, 3)
            },
            "gemini_api": {
                "category": gemini_category,
                "confidence": round(gemini_confidence, 3),
                "reasoning": gemini_reasoning
            },
            "input_text": text,
            "model_version": "hybrid_serverless_v1.1"
        }
    
    def _assess_risk(self, category: str, confidence: float) -> str:
        """Assess risk level"""
        if category in ['scam', 'phishing']:
            if confidence > 0.8:
                return "very_high"
            elif confidence > 0.6:
                return "high"
            else:
                return "medium"
        elif category == 'spam':
            if confidence > 0.8:
                return "medium"
            else:
                return "low"
        else:
            return "very_low"
    
    def get_model_stats(self) -> Dict:
        """Get basic model statistics"""
        return {
            "total_samples": 20,
            "categories": ["legitimate", "spam", "scam", "phishing"],
            "local_model_available": self.local_model is not None,
            "gemini_api_available": self.gemini_model is not None,
            "deployment": "serverless"
        }

# Global instance for serverless deployment
detector_instance = None

def get_detector():
    """Get or create detector instance"""
    global detector_instance
    if detector_instance is None:
        detector_instance = ServerlessHybridSpamDetector()
    return detector_instance
