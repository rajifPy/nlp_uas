import requests
import joblib
import os
from pathlib import Path

class HuggingFacePredictor:
    def __init__(self):
        self.model_url = os.getenv(
            'HF_MODEL_URL',
            'https://huggingface.co/murfhi/nlpuas/resolve/main/BEST_MODEL_LightGBM_TFIDF.joblib'
        )
        self.model = None
        self.model_cache_path = Path('/tmp/sdg_model.joblib')
        
        # Auto-load on init
        self._load_model()
    
    def _load_model(self):
        """Download dan load model dari HF"""
        try:
            # Check cache first
            if self.model_cache_path.exists():
                print("📦 Loading model from cache...")
                self.model = joblib.load(self.model_cache_path)
                print("✅ Model loaded from cache")
                return True
            
            # Download from HF
            print(f"⏬ Downloading model from Hugging Face...")
            response = requests.get(self.model_url, timeout=60)
            
            if response.status_code == 200:
                # Save to cache
                with open(self.model_cache_path, 'wb') as f:
                    f.write(response.content)
                
                # Load model
                self.model = joblib.load(self.model_cache_path)
                print("✅ Model downloaded and loaded successfully")
                return True
            else:
                print(f"❌ Download failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_sdgs(self, text: str, top_k: int = 3):
        """Predict using HF hosted model"""
        if not self.model:
            print("⚠️ Model not loaded, using fallback")
            return self._fallback_prediction(text, top_k)
        
        try:
            # Preprocess
            text = self._preprocess_text(text)
            
            # Predict (adjust based on your model structure)
            if isinstance(self.model, dict):
                vectorizer = self.model['vectorizer']
                classifier = self.model['model']
                features = vectorizer.transform([text])
                probs = classifier.predict_proba(features)[0]
            else:
                probs = self.model.predict_proba([text])[0]
            
            # Get top-k
            top_indices = probs.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if probs[idx] < 0.05:
                    continue
                
                results.append({
                    'sdg_number': int(idx + 1),
                    'sdg_name': self.sdg_labels[idx],
                    'confidence': float(probs[idx]),
                    'matched_keywords': self._extract_keywords(text, idx + 1),
                    'explanation': self._generate_explanation(idx + 1, probs[idx], text),
                    'source': 'huggingface_model'
                })
            
            return results if results else self._fallback_prediction(text, top_k)
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return self._fallback_prediction(text, top_k)
