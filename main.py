from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from hybrid_spam_detector_serverless import get_detector
import logging
from typing import Optional

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize hybrid detector (lazy loading for serverless)
detector = None

def get_detector_instance():
    global detector
    if detector is None:
        try:
            detector = get_detector()
            logger.info("✅ Serverless hybrid detector initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize detector: {str(e)}")
    return detector

app = FastAPI()

# Allow CORS for extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    text: str
    use_hybrid: Optional[bool] = True
    confidence_threshold: Optional[float] = 0.5

class TrainingRequest(BaseModel):
    text: str
    label: str
    category: Optional[str] = ""

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="စာသားမရှိပါ")
    
    detector = get_detector_instance()
    if not detector:
        raise HTTPException(status_code=500, detail="Detector မရရှိနိုင်ပါ")
    
    try:
        if req.use_hybrid:
            # Use hybrid approach
            result = detector.predict_hybrid(text)
            
            # Check if confidence meets threshold
            confidence = result["final_prediction"]["confidence"]
            if confidence < req.confidence_threshold:
                result["warning"] = "ယုံကြည်မှုအဆင့် နည်းပါသည်။ ထပ်မံစစ်ဆေးရန် အကြံပြုပါသည်။"
            
            return result
        else:
            # Use only Gemini API (legacy mode)
            gemini_category, gemini_confidence, gemini_reasoning = detector._predict_gemini(text)
            # Compute spam probability and binary label similar to hybrid
            spam_like = {"Spam", "Scam", "Phishing", "Malware", "Impersonation"}
            spam_probability = float(gemini_confidence if gemini_category in spam_like else (1.0 - gemini_confidence))
            binary_label = "spam" if spam_probability >= 0.5 else "general"

            return {
                "mode": "gemini_only",
                "category": gemini_category,
                "confidence": gemini_confidence,
                "reasoning": gemini_reasoning,
                "structured": getattr(detector, "_last_gemini_context", None),
                "binary_label": binary_label,
                "spam_probability": spam_probability,
                "spam_probability_pct": round(spam_probability * 100, 2)
            }
            
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"စစ်ဆေးရာတွင် အမှားရှိနေသည်: {str(e)}")

@app.post("/train")
async def add_training_data(req: TrainingRequest):
    """Add new training data to improve the model"""
    detector = get_detector_instance()
    if not detector:
        raise HTTPException(status_code=500, detail="Detector မရရှိနိုင်ပါ")
    
    try:
        # Note: Training data addition not supported in serverless mode
        # This would require persistent storage
        return {"message": "Serverless မုဒ်တွင် training data ထည့်သွင်းခြင်း မပံ့ပိုးပါ", "status": "not_supported"}
    except Exception as e:
        logger.error(f"Training data addition failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"လေ့ကျင့်ရေးဒေတာ ထည့်သွင်းရာတွင် အမှားရှိနေသည်: {str(e)}")

@app.get("/stats")
async def get_model_stats():
    """Get model statistics"""
    if not detector:
        raise HTTPException(status_code=500, detail="Detector မရရှိနိုင်ပါ")
    
    try:
        stats = detector.get_model_stats()
        return stats
    except Exception as e:
        logger.error(f"Stats retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"စာရင်းအင်း ရယူရာတွင် အမှားရှိနေသည်: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "detector_available": get_detector_instance() is not None,
        "local_model_available": get_detector_instance().local_model is not None if get_detector_instance() else False,
        "gemini_api_available": get_detector_instance().gemini_model is not None if get_detector_instance() else False,
        "message": "Serverless Hybrid Burmese Spam Detector API is running"
    }

@app.get("/")
def root():
    return {"message": "Hybrid Burmese Spam Detector API is running.", "version": "2.0-hybrid"} 
