# Hybrid Burmese Spam Detection System

A sophisticated spam detection system that combines **local machine learning models** with **Gemini API** to reduce hallucination and improve accuracy for Burmese language text analysis.

## 🚀 Key Features

### Hybrid Architecture
- **Local ML Model**: Trained on Burmese spam dataset using TF-IDF + Ensemble (Naive Bayes + Logistic Regression)
- **Gemini API Integration**: Provides contextual understanding and reasoning
- **Weighted Voting System**: Combines predictions with confidence scoring
- **Fallback Logic**: Graceful degradation when one component fails

### Advanced Capabilities
- **Real-time Training**: Add new samples to improve model performance
- **Confidence Thresholding**: Adjustable confidence levels for different use cases
- **Risk Assessment**: Multi-level risk categorization
- **Agreement Scoring**: Measures consensus between local and API models

## 📊 Detection Categories

| Category | Description | Risk Level |
|----------|-------------|------------|
| `legitimate` | Normal, safe messages | Very Low |
| `spam` | Promotional/marketing spam | Low-Medium |
| `scam` | Fraudulent schemes | High |
| `phishing` | Identity theft attempts | Very High |

## 🛠️ Installation & Setup

### 1. Install Dependencies
```bash
cd SpamDetector/backend
pip install -r requirements.txt
```

### 2. Environment Configuration
Create `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Initialize Dataset
The system comes with a pre-built Burmese spam dataset (`burmese_spam_dataset.csv`) containing:
- 20+ sample messages
- 4 categories (legitimate, spam, scam, phishing)
- Confidence scores and categorization

### 4. Start the API Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📡 API Endpoints

### POST `/analyze`
Analyze text for spam detection using hybrid approach.

**Request:**
```json
{
  "text": "သင့်အတွက် အထူးကမ်းလှမ်းချက်! ယခုပင် ဖုန်းနံပါတ်ကို ပေးပို့ပါ။",
  "use_hybrid": true,
  "confidence_threshold": 0.5
}
```

**Response:**
```json
{
  "final_prediction": {
    "category": "spam",
    "confidence": 0.892,
    "risk_level": "medium",
    "agreement": "high"
  },
  "local_model": {
    "category": "spam",
    "confidence": 0.856
  },
  "gemini_api": {
    "category": "spam",
    "confidence": 0.928,
    "reasoning": "This message contains promotional language..."
  },
  "input_text": "...",
  "model_version": "hybrid_v1.0"
}
```

### POST `/train`
Add new training data to improve model performance.

**Request:**
```json
{
  "text": "မနက်ဖန် အစည်းအဝေးရှိပါသည်။",
  "label": "legitimate",
  "category": "business"
}
```

### GET `/stats`
Get model statistics and performance metrics.

### GET `/health`
Check system health and component availability.

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_hybrid_system.py
```

This will test:
- ✅ Health check
- ✅ Hybrid analysis
- ✅ Gemini-only mode
- ✅ Training data addition
- ✅ Model statistics
- ✅ Performance comparison

## 🔧 Configuration Options

### Hybrid Prediction Weights
```python
local_weight = 0.6    # Local model influence
gemini_weight = 0.4   # Gemini API influence
```

### Model Parameters
```python
# TF-IDF Vectorizer
max_features = 1000
ngram_range = (1, 2)
min_df = 1
max_df = 0.95

# Ensemble Model
- MultinomialNB(alpha=0.1)
- LogisticRegression(max_iter=1000)
```

## 📈 Performance Benefits

### Reduced Hallucination
- Local model provides consistent baseline predictions
- Gemini API adds contextual understanding
- Disagreement detection flags uncertain cases

### Improved Accuracy
- Ensemble approach combines multiple algorithms
- Weighted voting considers model strengths
- Continuous learning from new data

### Reliability
- Graceful fallback when API is unavailable
- Local processing for privacy-sensitive data
- Confidence scoring for decision support

## 🔄 Model Training Workflow

1. **Initial Training**: Load CSV dataset and train local model
2. **Hybrid Prediction**: Combine local + Gemini predictions
3. **Feedback Loop**: Add new samples via `/train` endpoint
4. **Automatic Retraining**: Model updates with new data
5. **Performance Monitoring**: Track accuracy and agreement metrics

## 🛡️ Security & Privacy

- **Local Processing**: Sensitive data can be processed locally
- **API Fallback**: Gemini used only when needed
- **Confidence Thresholds**: Adjustable security levels
- **Risk Assessment**: Multi-level threat categorization

## 📝 Usage Examples

### Basic Spam Detection
```python
import requests

response = requests.post("http://localhost:8000/analyze", json={
    "text": "သင့်အား ငွေကြေးအကူအညီ ပေးနိုင်ပါသည်။",
    "use_hybrid": True
})

result = response.json()
print(f"Category: {result['final_prediction']['category']}")
print(f"Risk: {result['final_prediction']['risk_level']}")
```

### Adding Training Data
```python
requests.post("http://localhost:8000/train", json={
    "text": "ကျောင်းသားများအတွက် ပညာသင်ဆု",
    "label": "legitimate",
    "category": "education"
})
```

## 🚨 Troubleshooting

### Common Issues

1. **Gemini API Error**: Check API key in `.env` file
2. **Model Loading Failed**: Ensure CSV dataset exists
3. **Low Confidence**: Add more training data for specific categories
4. **Disagreement**: Review edge cases and add targeted samples

### Debug Mode
Set logging level to DEBUG for detailed information:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 🔮 Future Enhancements

- [ ] Multi-language support expansion
- [ ] Real-time model updates
- [ ] Advanced ensemble techniques
- [ ] Custom confidence calibration
- [ ] Integration with other LLM providers
- [ ] Automated dataset expansion

---

**Version**: 2.0-hybrid  
**Last Updated**: 2025-08-21  
**Compatibility**: Python 3.8+, FastAPI 0.104+
