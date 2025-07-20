from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-2.0-flash")

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

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    text = req.text.strip()
    if not text:
        return {"error": "စာသားမရှိပါ"}
    prompt = f"""
    အောက်ပါ မြန်မာစကားပြောကို စစ်ဆေးပါ။
    အမျိုးအစား: ယုံကြည်စိတ်ချရ/စပမ်/လှည်ဖြားမှု/ဖစ်ရှင်း
    ရလဒ်နှင့်ရှင်းပြချက်ကို မြန်မာဘာသာဖြင့်ပေးပါ။
    စကားပြော: '''{text}'''
    """
    try:
        response = model.generate_content(prompt)
        return {"result": response.text}
    except Exception as e:
        return {"error": f"AI အမှား: {str(e)}"}

@app.get("/")
def root():
    return {"message": "Burmese Spam Detector API is running."} 
