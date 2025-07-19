from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
import re

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

def is_cyber_hygiene_question(text):
    """Check if the text is asking about cyber hygiene, security, or general cybersecurity questions"""
    cyber_keywords = [
        'password', 'security', 'cyber', 'hygiene', 'protection', 'safe', 'secure',
        'virus', 'malware', 'hack', 'breach', 'privacy', 'data', 'encryption',
        'firewall', 'antivirus', 'backup', 'update', 'patch', 'vulnerability',
        'phishing', 'social engineering', 'two factor', '2fa', 'authentication',
        'password manager', 'vpn', 'network security', 'online safety',
        'digital security', 'internet safety', 'cybersecurity', 'cyber security',
        'ပါစ်ဝပ်', 'လုံခြုံရေး', 'ဆိုက်ဘာ', 'ဗိုင်းရပ်စ်', 'ဟက်ခ်', 'ဒေတာ',
        'အီးမေးလ်', 'အင်တာနက်', 'အွန်လိုင်း', 'အန္တရာယ်', 'ကာကွယ်ရေး',
        'အချက်အလက်', 'လျှို့ဝှက်ချက်', 'အကောင့်', 'ဘဏ်', 'ငွေကြေး'
    ]
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in cyber_keywords)

def is_general_educational_text(text):
    """Check if the text is general educational/informational content that shouldn't be analyzed for spam"""
    # Only trigger for clear definition/explanation patterns or medical/health info
    text_lower = text.lower()
    # Definition/explanation patterns
    definition_patterns = [
        'definition of', 'what is', 'what are', 'explain', 'describe',
        'အဓိပ္ပာယ်', 'ဆိုလိုရင်း', 'ဘာလဲ', 'ဘာတွေလဲ', 'ရှင်းပြပါ', 'ဖော်ပြပါ',
    ]
    if any(pattern in text_lower for pattern in definition_patterns):
        return True
    # Medical/health content
    medical_keywords = ['hallucination', 'hallucinations', 'medical', 'health', 'doctor', 'treatment', 'symptoms',
        'ဆေးပညာ', 'ကျန်းမာရေး', 'လက္ခဏာ', 'ကုသမှု', 'ရောဂါ', 'အခြေအနေ']
    if any(keyword in text_lower for keyword in medical_keywords):
        return True
    # Obvious English/Burmese definition structure
    if 'is where' in text_lower or 'are where' in text_lower or 'means' in text_lower:
        return True
    return False

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    text = req.text.strip()
    if not text:
        return {"error": "စာသားမရှိပါ"}
    
    # Check if this is general educational text
    if is_general_educational_text(text):
        return {
            "result": """ဤစာသားသည် ပညာရေးဆိုင်ရာ သို့မဟုတ် အချက်အလက်ပေးသော စာသားဖြစ်ပါသည်။

ကျွန်ုပ်တို့၏ Spam Detector သည် မက်ဆေ့ချ်များ၊ အီးမေးလ်များ သို့မဟုတ် ဆက်သွယ်ရေးစာသားများကို စစ်ဆေးရန် ဒီဇိုင်းပြုလုပ်ထားပါသည်။

ပညာရေးဆိုင်ရာ သို့မဟုတ် အချက်အလက်ပေးသော စာသားများအတွက် သင့်အား ပိုမိုကောင်းမွန်သော အကူအညီပေးနိုင်ရန် ကျွန်ုပ်တို့၏ ဆိုက်ဘာလုံခြုံရေး ချက်တ်ဘော့ကို အသုံးပြုပါ။"""
        }
    
    # Check if this is a cyber hygiene question
    if is_cyber_hygiene_question(text):
        # Provide a brief, direct answer based on the question type
        if any(word in text.lower() for word in ['cyber law', 'cyberlaw', 'ဆိုက်ဘာဥပဒေ', 'ဥပဒေ']):
            answer = "ဆိုက်ဘာဥပဒေသည် အင်တာနက်နှင့် ဒစ်ဂျစ်တယ်နည်းပညာဆိုင်ရာ ဥပဒေများဖြစ်ပါသည်။"
        elif any(word in text.lower() for word in ['password', 'ပါစ်ဝပ်', 'စကားဝှက်']):
            answer = "စကားဝှက်များသည် သင့်အကောင့်များကို ကာကွယ်ရန် အရေးကြီးသော လုံခြုံရေးနည်းလမ်းတစ်ခုဖြစ်ပါသည်။"
        elif any(word in text.lower() for word in ['virus', 'malware', 'ဗိုင်းရပ်စ်', 'အန္တရာယ်']):
            answer = "ဗိုင်းရပ်စ်နှင့် malware များသည် ကွန်ပျူတာနှင့် ဒေတာများကို အန္တရာယ်ပြုနိုင်သော ဆော့ဖ်ဝဲများဖြစ်ပါသည်။"
        elif any(word in text.lower() for word in ['phishing', 'ဖစ်ရှင်း', 'လှည်ဖြား']):
            answer = "Phishing သည် သင့်ကို လှည်ဖြားပြီး လျှို့ဝှက်အချက်အလက်များ ရယူရန် ကြိုးစားသော နည်းလမ်းတစ်ခုဖြစ်ပါသည်။"
        elif any(word in text.lower() for word in ['security', 'လုံခြုံရေး', 'ကာကွယ်ရေး']):
            answer = "အွန်လိုင်းလုံခြုံရေးသည် သင့်ဒေတာနှင့် ကိုယ်ရေးကိုယ်တာကို ကာကွယ်ရန် အရေးကြီးပါသည်။"
        else:
            answer = "ဤမေးခွန်းသည် ဆိုက်ဘာလုံခြုံရေးနှင့် သက်ဆိုင်ပါသည်။"
        
        return {
            "result": f"""{answer}

ပိုမိုအသေးစိတ်သော အကူအညီအတွက် ကျွန်ုပ်တို့၏ ဆိုက်ဘာလုံခြုံရေး ချက်တ်ဘော့ကို အသုံးပြုပါ။

ဘေးဘက်မီနူးတွင် "Cyber law hygiene chat" ကို ရွေးချယ်ပါ။"""
        }
    
    # Enhanced prompt to prevent hallucinations and be more conservative
    prompt = f"""
    အောက်ပါ မြန်မာစကားပြောကို စစ်ဆေးပါ။
    
    အရေးကြီးသော လမ်းညွှန်ချက်များ:
    1. သံသယရှိသော သို့မဟုတ် အန္တရာယ်ရှိသော စကားပြောများကိုသာ စပမ်/လှည်ဖြားမှု/ဖစ်ရှင်း အဖြစ် ခွဲခြားပါ
    2. ရိုးရှင်းသော သို့မဟုတ် ယုံကြည်စိတ်ချရသော စကားပြောများကို "ယုံကြည်စိတ်ချရ" အဖြစ် ခွဲခြားပါ
    3. မရေရာသော သို့မဟုတ် သံသယရှိသော အခြေအနေများတွင် "သတိထားရန်" သို့မဟုတ် "ထပ်မံစစ်ဆေးရန်" ဟု အကြံပြုပါ
    4. မည်သည့်အခါမျှ "အန္တရာယ်မရှိ" သို့မဟုတ် "လုံခြုံသည်" ဟု အတည်ပြုမပြောပါနှင့်
    5. သံသယရှိသော အချက်များကို အသေးစိတ်ရှင်းပြပါ
    
    ခွဲခြားမှုရလဒ်:
    - ယုံကြည်စိတ်ချရ (Legitimate)
    - စပမ် (Spam) 
    - လှည်ဖြားမှု (Scam)
    - ဖစ်ရှင်း (Phishing)
    - သတိထားရန် (Caution)
    
    စကားပြော: '''{text}'''
    
    ကျေးဇူးပြု၍ ခွဲခြားမှုရလဒ်၊ ရှင်းပြချက်နှင့် အကြံပြုချက်များကို မြန်မာဘာသာဖြင့် ပေးပါ။
    """
    
    try:
        response = model.generate_content(prompt)
        return {"result": response.text}
    except Exception as e:
        return {"error": f"AI အမှား: {str(e)}"}

@app.get("/")
def root():
    return {"message": "Burmese Spam Detector API is running."} 
