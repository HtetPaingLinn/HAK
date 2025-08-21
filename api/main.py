from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from detector_light import get_detector
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
            return {
                "category": gemini_category,
                "confidence": gemini_confidence,
                "reasoning": gemini_reasoning,
                "mode": "gemini_only"
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
    det = get_detector_instance()
    if not det:
        raise HTTPException(status_code=500, detail="Detector မရရှိနိုင်ပါ")
    try:
        stats = det.get_model_stats()
        return stats
    except Exception as e:
        logger.error(f"Stats retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"စာရင်းအင်း ရယူရာတွင် အမှားရှိနေသည်: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    det = get_detector_instance()
    local_available = False
    gemini_available = False
    if det is not None:
        try:
            local_available = bool(getattr(det, "local_model", False))
            gemini_available = getattr(det, "gemini_model", None) is not None
        except Exception:
            local_available = False
            gemini_available = False
    return {
        "status": "healthy",
        "detector_available": det is not None,
        "local_model_available": local_available,
        "gemini_api_available": gemini_available,
        "message": "Serverless Hybrid Burmese Spam Detector API is running"
    }

@app.get("/")
def root():
    return {"message": "Hybrid Burmese Spam Detector API is running.", "version": "2.0-hybrid"}
