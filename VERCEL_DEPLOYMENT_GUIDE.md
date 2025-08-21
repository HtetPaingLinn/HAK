# Vercel Deployment Guide - Hybrid Burmese Spam Detector

## 🚀 Quick Deployment Steps

### 1. **Environment Variables Setup**
In your Vercel dashboard, add the following environment variable:
```
GEMINI_API_KEY = your_actual_gemini_api_key_here
```

### 2. **Git Push Workflow**
```bash
# Navigate to your backend directory
cd SpamDetector/backend

# Add all changes
git add .

# Commit changes
git commit -m "Deploy hybrid spam detector to Vercel"

# Push to your repository
git push origin main
```

### 3. **Vercel Auto-Deploy**
- Vercel will automatically detect the push and start deployment
- The `vercel.json` configuration will handle the serverless setup
- Deployment typically takes 2-3 minutes

## 📁 **Files Optimized for Vercel**

### Core Files:
- `main.py` - FastAPI application with lazy loading
- `hybrid_spam_detector_serverless.py` - Serverless-optimized detector
- `vercel.json` - Deployment configuration
- `requirements.txt` - Python dependencies

### Key Optimizations:
- **Lazy Loading**: Detector initializes only when needed
- **Embedded Dataset**: CSV data included in code (no file I/O)
- **Caching**: LRU cache for local predictions
- **Reduced Model Size**: Smaller TF-IDF features (500 vs 1000)
- **Timeout Handling**: Optimized for serverless constraints

## 🔧 **Vercel Configuration**

Your `vercel.json` is configured with:
```json
{
  "version": 2,
  "builds": [
    { "src": "main.py", "use": "@vercel/python" }
  ],
  "routes": [
    { "src": "/(.*)", "dest": "main.py" }
  ],
  "functions": {
    "main.py": {
      "maxDuration": 30
    }
  },
  "env": {
    "GEMINI_API_KEY": "@gemini_api_key"
  }
}
```

## 📡 **API Endpoints (Live)**

Once deployed, your API will be available at:
`https://your-project-name.vercel.app`

### Available Endpoints:
- `GET /` - API status
- `GET /health` - System health check
- `POST /analyze` - Hybrid spam analysis
- `GET /stats` - Model statistics
- `POST /train` - Training endpoint (disabled in serverless)

## 🧪 **Testing Your Deployment**

### 1. Health Check
```bash
curl https://your-project-name.vercel.app/health
```

### 2. Spam Analysis
```bash
curl -X POST https://your-project-name.vercel.app/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "သင့်အတွက် အထူးကမ်းလှမ်းချက်! ယခုပင် ဖုန်းနံပါတ်ကို ပေးပို့ပါ။",
    "use_hybrid": true,
    "confidence_threshold": 0.5
  }'
```

## 🔄 **Update Workflow**

### For Code Changes:
1. Make changes to your files
2. Test locally (optional)
3. Git add, commit, and push
4. Vercel auto-deploys

### For Environment Variables:
1. Go to Vercel Dashboard
2. Select your project
3. Go to Settings → Environment Variables
4. Update `GEMINI_API_KEY`
5. Redeploy if needed

## 📊 **Performance Considerations**

### Cold Start Optimization:
- Singleton pattern for detector instance
- Lazy loading of ML models
- Cached predictions for repeated queries
- Embedded dataset (no file system reads)

### Memory Usage:
- Reduced TF-IDF features: 500 (vs 1000 local)
- Lightweight ensemble model
- Efficient text preprocessing
- LRU cache with 100 item limit

## 🛠️ **Troubleshooting**

### Common Issues:

1. **Deployment Fails**
   - Check `requirements.txt` for version conflicts
   - Verify `vercel.json` syntax
   - Check build logs in Vercel dashboard

2. **Gemini API Errors**
   - Verify `GEMINI_API_KEY` in environment variables
   - Check API key permissions and quotas
   - Monitor function timeout (30s max)

3. **Model Loading Issues**
   - Check function memory limits
   - Verify embedded dataset format
   - Review cold start performance

### Debug Commands:
```bash
# Check deployment status
vercel ls

# View function logs
vercel logs your-project-name

# Local testing
vercel dev
```

## 🔐 **Security Best Practices**

- ✅ API key stored as environment variable
- ✅ CORS configured for your domains
- ✅ No sensitive data in code
- ✅ Serverless isolation
- ✅ HTTPS by default

## 📈 **Monitoring & Analytics**

### Vercel Dashboard Metrics:
- Function invocations
- Response times
- Error rates
- Bandwidth usage

### Custom Monitoring:
- Add logging for prediction accuracy
- Track confidence score distributions
- Monitor API response times
- Set up alerts for high error rates

## 🔮 **Next Steps**

After successful deployment:

1. **Update Chrome Extension**: Point to new Vercel URL
2. **Test Integration**: Verify extension → API communication
3. **Monitor Performance**: Check response times and accuracy
4. **Scale if Needed**: Upgrade Vercel plan for higher limits

## 📞 **Support**

If you encounter issues:
1. Check Vercel function logs
2. Verify environment variables
3. Test API endpoints manually
4. Review this guide for common solutions

---

**Deployment Version**: Serverless Hybrid v1.0  
**Last Updated**: 2025-08-21  
**Vercel Compatibility**: ✅ Optimized
