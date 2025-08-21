#!/usr/bin/env python3
"""
Test script for the Hybrid Burmese Spam Detection System
"""

import requests
import json
import time
from typing import Dict, List

class HybridSystemTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def test_health_check(self) -> bool:
        """Test if the API is running"""
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print("🟢 Health Check:")
                print(f"   Status: {data['status']}")
                print(f"   Detector Available: {data['detector_available']}")
                print(f"   Local Model: {data['local_model_available']}")
                print(f"   Gemini API: {data['gemini_api_available']}")
                return True
            else:
                print(f"🔴 Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"🔴 Health check error: {str(e)}")
            return False
    
    def test_hybrid_analysis(self, test_messages: List[str]) -> List[Dict]:
        """Test hybrid spam analysis"""
        results = []
        print("\n🔍 Testing Hybrid Analysis:")
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n📝 Test {i}: {message[:50]}...")
            
            try:
                payload = {
                    "text": message,
                    "use_hybrid": True,
                    "confidence_threshold": 0.5
                }
                
                response = requests.post(f"{self.base_url}/analyze", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    results.append(result)
                    
                    # Display results
                    final = result["final_prediction"]
                    local = result["local_model"]
                    gemini = result["gemini_api"]
                    
                    print(f"   🎯 Final: {final['category']} (confidence: {final['confidence']:.3f})")
                    print(f"   🤖 Local: {local['category']} (confidence: {local['confidence']:.3f})")
                    print(f"   🧠 Gemini: {gemini['category']} (confidence: {gemini['confidence']:.3f})")
                    print(f"   ⚠️ Risk Level: {final['risk_level']}")
                    print(f"   🤝 Agreement: {final['agreement']}")
                    
                    if "warning" in result:
                        print(f"   ⚠️ Warning: {result['warning']}")
                        
                else:
                    print(f"   🔴 Analysis failed: {response.status_code}")
                    print(f"   Error: {response.text}")
                    
            except Exception as e:
                print(f"   🔴 Analysis error: {str(e)}")
        
        return results
    
    def test_gemini_only_mode(self, test_messages: List[str]) -> List[Dict]:
        """Test Gemini-only analysis mode"""
        results = []
        print("\n🧠 Testing Gemini-Only Mode:")
        
        for i, message in enumerate(test_messages[:3], 1):  # Test fewer for comparison
            print(f"\n📝 Test {i}: {message[:50]}...")
            
            try:
                payload = {
                    "text": message,
                    "use_hybrid": False,
                    "confidence_threshold": 0.5
                }
                
                response = requests.post(f"{self.base_url}/analyze", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    results.append(result)
                    
                    print(f"   🧠 Gemini: {result['category']} (confidence: {result['confidence']:.3f})")
                    print(f"   📝 Reasoning: {result['reasoning'][:100]}...")
                    
                else:
                    print(f"   🔴 Analysis failed: {response.status_code}")
                    
            except Exception as e:
                print(f"   🔴 Analysis error: {str(e)}")
        
        return results
    
    def test_training_data_addition(self) -> bool:
        """Test adding new training data"""
        print("\n📚 Testing Training Data Addition:")
        
        new_training_samples = [
            {
                "text": "သင့်အား ငွေကြေးအကူအညီ ပေးနိုင်ပါသည်။ ချက်ချင်း ဆက်သွယ်ပါ။",
                "label": "scam",
                "category": "financial"
            },
            {
                "text": "ကျောင်းသားများအတွက် ပညာသင်ဆု လျှောက်ထားနိုင်ပါသည်။",
                "label": "legitimate", 
                "category": "education"
            }
        ]
        
        for i, sample in enumerate(new_training_samples, 1):
            print(f"\n📝 Adding sample {i}: {sample['text'][:50]}...")
            
            try:
                response = requests.post(f"{self.base_url}/train", json=sample)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ {result['message']}")
                else:
                    print(f"   🔴 Training failed: {response.status_code}")
                    return False
                    
            except Exception as e:
                print(f"   🔴 Training error: {str(e)}")
                return False
        
        return True
    
    def test_model_statistics(self) -> Dict:
        """Test model statistics endpoint"""
        print("\n📊 Testing Model Statistics:")
        
        try:
            response = requests.get(f"{self.base_url}/stats")
            
            if response.status_code == 200:
                stats = response.json()
                print(f"   📈 Total Samples: {stats.get('total_samples', 0)}")
                print(f"   📋 Label Distribution: {stats.get('label_distribution', {})}")
                print(f"   🏷️ Category Distribution: {stats.get('category_distribution', {})}")
                print(f"   🤖 Local Model Available: {stats.get('local_model_available', False)}")
                print(f"   🧠 Gemini API Available: {stats.get('gemini_api_available', False)}")
                return stats
            else:
                print(f"   🔴 Stats failed: {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"   🔴 Stats error: {str(e)}")
            return {}
    
    def run_comprehensive_test(self):
        """Run all tests"""
        print("🚀 Starting Comprehensive Hybrid System Test")
        print("=" * 60)
        
        # Test messages in Burmese
        test_messages = [
            "သင့်အတွက် အထူးကမ်းလှမ်းချက်! ယခုပင် ဖုန်းနံပါတ်ကို ပေးပို့ပါ။",  # spam
            "မနက်ဖန် ရုံးမှာ တွေ့ကြမယ်နော်။ စာရွက်စာတမ်းတွေ မမေ့နဲ့။",  # legitimate
            "ဘဏ်အကောင့်ကို အတည်ပြုရန် လိုအပ်ပါသည်။ လင့်ခ်ကို နှိပ်ပါ။",  # phishing
            "သင်သည် ၁သိန်းကျပ် အနိုင်ရရှိပါပြီ! ချက်ချင်း ဆက်သွယ်ပါ။",  # spam
            "ဒီနေ့ ညနေ ၆နာရီမှာ အိမ်ပြန်မယ်။ ညစာ ပြင်ထားပါ။",  # legitimate
            "သင့်ဖုန်းကို ဗိုင်းရပ်စ်ကူးစက်ထားပါသည်။ ချက်ချင်းဒေါင်းလုဒ်လုပ်ပါ။"  # scam
        ]
        
        # 1. Health Check
        if not self.test_health_check():
            print("🔴 System not ready. Exiting...")
            return
        
        # 2. Test Hybrid Analysis
        hybrid_results = self.test_hybrid_analysis(test_messages)
        
        # 3. Test Gemini-only Mode
        gemini_results = self.test_gemini_only_mode(test_messages)
        
        # 4. Test Training Data Addition
        training_success = self.test_training_data_addition()
        
        # 5. Test Model Statistics
        stats = self.test_model_statistics()
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 Test Summary:")
        print(f"   ✅ Health Check: Passed")
        print(f"   ✅ Hybrid Analysis: {len(hybrid_results)} tests completed")
        print(f"   ✅ Gemini-only Mode: {len(gemini_results)} tests completed")
        print(f"   {'✅' if training_success else '🔴'} Training Data: {'Passed' if training_success else 'Failed'}")
        print(f"   ✅ Model Statistics: {'Available' if stats else 'Failed'}")
        
        # Performance comparison
        if hybrid_results and gemini_results:
            print("\n🔄 Performance Comparison (Hybrid vs Gemini-only):")
            for i in range(min(len(hybrid_results), len(gemini_results))):
                hybrid = hybrid_results[i]["final_prediction"]
                gemini = gemini_results[i]
                
                print(f"   Test {i+1}:")
                print(f"     Hybrid: {hybrid['category']} ({hybrid['confidence']:.3f})")
                print(f"     Gemini: {gemini['category']} ({gemini['confidence']:.3f})")
                print(f"     Agreement: {hybrid['category'] == gemini['category']}")

if __name__ == "__main__":
    # Initialize tester
    tester = HybridSystemTester()
    
    # Run comprehensive test
    tester.run_comprehensive_test()
