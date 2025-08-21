import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import google.generativeai as genai
import os
from dotenv import load_dotenv
import re
import logging
from typing import Dict, Tuple, List
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridBurmeseSpamDetector:
    def __init__(self, csv_path: str = "burmese_spam_dataset.csv"):
        """
        Initialize hybrid spam detector with local ML model and Gemini API
        """
        load_dotenv()
        self.csv_path = csv_path
        self.vectorizer = None
        self.local_model = None
        self.gemini_model = None
        self.label_mapping = {
            'legitimate': 0,
            'spam': 1, 
            'scam': 2,
            'phishing': 3
        }
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        
        # Initialize Gemini
        self._setup_gemini()
        
        # Load and train local model
        self._load_and_train_local_model()
    
    def _setup_gemini(self):
        """Setup Gemini API"""
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")
                logger.info("✅ Gemini API initialized successfully")
            else:
                logger.warning("⚠️ GEMINI_API_KEY not found")
        except Exception as e:
            logger.error(f"❌ Gemini setup failed: {str(e)}")
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess Burmese text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep Burmese characters
        text = re.sub(r'[^\u1000-\u109F\u1040-\u1049\s\w]', ' ', text)
        return text.lower()
    
    def _load_and_train_local_model(self):
        """Load CSV data and train local ML model"""
        try:
            # Load dataset
            df = pd.read_csv(self.csv_path)
            logger.info(f"📊 Loaded {len(df)} samples from dataset")
            
            # Preprocess texts
            df['processed_text'] = df['text'].apply(self._preprocess_text)
            
            # Map labels to numbers
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
                max_features=1000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            
            # Fit vectorizer and transform data
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            X_test_tfidf = self.vectorizer.transform(X_test)
            
            # Create ensemble model
            nb_model = MultinomialNB(alpha=0.1)
            lr_model = LogisticRegression(random_state=42, max_iter=1000)
            
            self.local_model = VotingClassifier(
                estimators=[
                    ('naive_bayes', nb_model),
                    ('logistic_regression', lr_model)
                ],
                voting='soft'
            )
            
            # Train model
            self.local_model.fit(X_train_tfidf, y_train)
            
            # Evaluate model
            y_pred = self.local_model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"🎯 Local model accuracy: {accuracy:.3f}")
            logger.info("✅ Local model trained successfully")
            
            # Save model and vectorizer
            self._save_model()
            
        except Exception as e:
            logger.error(f"❌ Local model training failed: {str(e)}")
    
    def _save_model(self):
        """Save trained model and vectorizer"""
        try:
            joblib.dump(self.local_model, 'local_spam_model.pkl')
            joblib.dump(self.vectorizer, 'tfidf_vectorizer.pkl')
            logger.info("💾 Model saved successfully")
        except Exception as e:
            logger.error(f"❌ Model saving failed: {str(e)}")
    
    def _load_model(self):
        """Load saved model and vectorizer"""
        try:
            self.local_model = joblib.load('local_spam_model.pkl')
            self.vectorizer = joblib.load('tfidf_vectorizer.pkl')
            logger.info("📂 Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Model loading failed: {str(e)}")
            return False
    
    def _predict_local(self, text: str) -> Tuple[str, float]:
        """Predict using local ML model"""
        try:
            if not self.local_model or not self.vectorizer:
                return "unknown", 0.0
            
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Vectorize
            text_tfidf = self.vectorizer.transform([processed_text])
            
            # Predict
            prediction = self.local_model.predict(text_tfidf)[0]
            probabilities = self.local_model.predict_proba(text_tfidf)[0]
            confidence = max(probabilities)
            
            label = self.reverse_label_mapping.get(prediction, "unknown")
            
            return label, confidence
            
        except Exception as e:
            logger.error(f"❌ Local prediction failed: {str(e)}")
            return "unknown", 0.0
    
    def _predict_gemini(self, text: str) -> Tuple[str, float, str]:
        """Predict using Gemini API"""
        try:
            if not self.gemini_model:
                return "unknown", 0.0, "Gemini API not available"
            
            prompt = f"""
            အောက်ပါ မြန်မာစကားပြောကို စစ်ဆေးပြီး JSON format ဖြင့် ပြန်ပေးပါ။
            
            ရလဒ်တွင် အောက်ပါအချက်များ ပါဝင်ရမည်:
            1. "category": "legitimate", "spam", "scam", သို့မဟုတ် "phishing"
            2. "confidence": 0.0 မှ 1.0 အထိ
            3. "reasoning": အတိုချုပ်ရှင်းပြချက်
            
            စကားပြော: "{text}"
            
            JSON format:
            {{
                "category": "...",
                "confidence": 0.0,
                "reasoning": "..."
            }}
            """
            
            response = self.gemini_model.generate_content(prompt)
            
            # Parse JSON response
            try:
                # Extract JSON from response
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
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"❌ JSON parsing failed: {str(e)}")
                # Fallback: extract category from text
                response_lower = response.text.lower()
                if 'spam' in response_lower:
                    return 'spam', 0.7, response.text
                elif 'scam' in response_lower:
                    return 'scam', 0.7, response.text
                elif 'phishing' in response_lower:
                    return 'phishing', 0.7, response.text
                else:
                    return 'legitimate', 0.6, response.text
            
        except Exception as e:
            logger.error(f"❌ Gemini prediction failed: {str(e)}")
            return "unknown", 0.0, f"Error: {str(e)}"
    
    def predict_hybrid(self, text: str) -> Dict:
        """
        Hybrid prediction combining local ML and Gemini API
        """
        # Get local prediction
        local_category, local_confidence = self._predict_local(text)
        
        # Get Gemini prediction
        gemini_category, gemini_confidence, gemini_reasoning = self._predict_gemini(text)
        
        # Combine predictions with weighted voting
        local_weight = 0.6  # Give more weight to local model for consistency
        gemini_weight = 0.4
        
        # Calculate final prediction
        if local_category == gemini_category:
            # Both models agree
            final_category = local_category
            final_confidence = (local_confidence * local_weight + gemini_confidence * gemini_weight)
            agreement = "high"
        else:
            # Models disagree - use higher confidence
            if local_confidence > gemini_confidence:
                final_category = local_category
                final_confidence = local_confidence * 0.8  # Reduce confidence due to disagreement
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
            "model_version": "hybrid_v1.0"
        }
    
    def _assess_risk(self, category: str, confidence: float) -> str:
        """Assess risk level based on category and confidence"""
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
        else:  # legitimate
            return "very_low"
    
    def add_training_data(self, text: str, label: str, category: str = "", confidence: float = 1.0):
        """Add new training data to CSV"""
        try:
            # Append to CSV
            new_data = pd.DataFrame({
                'text': [text],
                'label': [label],
                'category': [category],
                'confidence': [confidence]
            })
            
            new_data.to_csv(self.csv_path, mode='a', header=False, index=False)
            logger.info(f"✅ Added new training data: {label}")
            
            # Retrain model with new data
            self._load_and_train_local_model()
            
        except Exception as e:
            logger.error(f"❌ Failed to add training data: {str(e)}")
    
    def get_model_stats(self) -> Dict:
        """Get model statistics"""
        try:
            df = pd.read_csv(self.csv_path)
            stats = {
                "total_samples": len(df),
                "label_distribution": df['label'].value_counts().to_dict(),
                "category_distribution": df['category'].value_counts().to_dict(),
                "local_model_available": self.local_model is not None,
                "gemini_api_available": self.gemini_model is not None
            }
            return stats
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {str(e)}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    # Initialize detector
    detector = HybridBurmeseSpamDetector()
    
    # Test messages
    test_messages = [
        "သင့်အတွက် အထူးကမ်းလှမ်းချက်! ယခုပင် ဖုန်းနံပါတ်ကို ပေးပို့ပါ။",
        "မနက်ဖန် ရုံးမှာ တွေ့ကြမယ်နော်။ စာရွက်စာတမ်းတွေ မမေ့နဲ့။",
        "ဘဏ်အကောင့်ကို အတည်ပြုရန် လိုအပ်ပါသည်။ လင့်ခ်ကို နှိပ်ပါ။"
    ]
    
    # Test predictions
    for message in test_messages:
        print(f"\n📝 Testing: {message}")
        result = detector.predict_hybrid(message)
        print(f"🎯 Result: {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    # Print model stats
    print(f"\n📊 Model Statistics:")
    stats = detector.get_model_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
