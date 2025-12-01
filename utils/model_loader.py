import joblib
import os
from pathlib import Path
import numpy as np
import re

class ModelLoader:
    """Loader untuk LightGBM model (.joblib format)"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path or os.getenv('MODEL_PATH', './models')
        self.sdg_model = None
        self.vectorizer = None
        self.sdg_labels = [
            'No Poverty', 'Zero Hunger', 'Good Health and Well-being',
            'Quality Education', 'Gender Equality', 'Clean Water and Sanitation',
            'Affordable and Clean Energy', 'Decent Work and Economic Growth',
            'Industry, Innovation and Infrastructure', 'Reduced Inequality',
            'Sustainable Cities and Communities', 'Responsible Consumption and Production',
            'Climate Action', 'Life Below Water', 'Life on Land',
            'Peace, Justice and Strong Institutions', 'Partnerships for the Goals'
        ]
        
        self.load_models()
    
    def load_models(self):
        """Load LightGBM model dari .joblib file"""
        try:
            print(f"🔍 Looking for model in: {self.model_path}")
            
            # Cek path model
            model_path = Path(self.model_path) / 'BEST_MODEL_LightGBM_TFIDF.joblib'
            
            if not model_path.exists():
                # Try alternative paths
                alt_paths = [
                    Path('/tmp/models/BEST_MODEL_LightGBM_TFIDF.joblib'),
                    Path('./models/BEST_MODEL_LightGBM_TFIDF.joblib'),
                    Path('BEST_MODEL_LightGBM_TFIDF.joblib')
                ]
                
                for alt_path in alt_paths:
                    if alt_path.exists():
                        model_path = alt_path
                        break
            
            if model_path.exists():
                print(f"📥 Loading model from: {model_path}")
                
                # Load model (bisa berupa pipeline atau dict)
                loaded = joblib.load(model_path)
                
                # Check if it's a pipeline atau dict
                if isinstance(loaded, dict):
                    self.sdg_model = loaded.get('model')
                    self.vectorizer = loaded.get('vectorizer')
                    print("✅ Loaded model from dictionary")
                elif hasattr(loaded, 'predict_proba'):
                    # It's a sklearn Pipeline atau model langsung
                    self.sdg_model = loaded
                    print("✅ Loaded model directly")
                else:
                    print("❌ Unknown model format")
                    self.sdg_model = None
                
                print(f"✅ Model loaded! Type: {type(self.sdg_model)}")
                
            else:
                print(f"❌ Model not found at: {model_path}")
                print("💡 Please upload BEST_MODEL_LightGBM_TFIDF.joblib to /tmp/models/")
                self.sdg_model = None
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            self.sdg_model = None
    
    def predict_sdgs(self, text: str, top_k: int = 3):
        """
        Predict SDGs menggunakan LightGBM model
        """
        if not self.sdg_model:
            print("❌ Model not loaded, using fallback")
            return self._fallback_prediction(text, top_k)
        
        try:
            print(f"🔍 Analyzing text...")
            
            # Preprocessing
            processed_text = self._preprocess_text(text)
            
            # Predict
            # Jika model adalah pipeline, langsung predict
            if hasattr(self.sdg_model, 'predict_proba'):
                probabilities = self.sdg_model.predict_proba([processed_text])[0]
            # Jika vectorizer terpisah
            elif self.vectorizer:
                features = self.vectorizer.transform([processed_text])
                probabilities = self.sdg_model.predict_proba(features)[0]
            else:
                print("❌ Cannot determine how to use model")
                return self._fallback_prediction(text, top_k)
            
            print(f"📈 Prediction complete")
            
            # Get top-k
            top_indices = np.argsort(probabilities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                prob = probabilities[idx]
                
                if prob < 0.05:  # Skip very low confidence
                    continue
                
                sdg_number = idx + 1
                sdg_name = self.sdg_labels[idx]
                
                results.append({
                    'sdg_number': sdg_number,
                    'sdg_name': sdg_name,
                    'confidence': float(prob),
                    'matched_keywords': self._extract_keywords(text, sdg_number),
                    'explanation': self._generate_explanation(sdg_number, prob, text)
                })
            
            if not results:
                return self._fallback_prediction(text, top_k)
            
            return results
            
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._fallback_prediction(text, top_k)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocessing text"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _generate_explanation(self, sdg_number: int, confidence: float, text: str) -> str:
        """Generate explanation"""
        if confidence >= 0.7:
            level = "Strong alignment"
        elif confidence >= 0.4:
            level = "Moderate alignment"
        else:
            level = "Weak alignment"
        
        keywords = self._extract_keywords(text, sdg_number)[:3]
        kw_text = ', '.join(keywords) if keywords else "general terms"
        
        return f"{level} with SDG {sdg_number}. Key indicators: {kw_text}"
    
    def _extract_keywords(self, text: str, sdg_number: int) -> list:
        """Extract keywords untuk SDG tertentu"""
        sdg_keywords = {
            1: ['poverty', 'poor', 'income', 'economic', 'financial'],
            2: ['hunger', 'food', 'nutrition', 'agriculture', 'farming'],
            3: ['health', 'medical', 'disease', 'healthcare', 'wellbeing'],
            4: ['education', 'school', 'learning', 'student', 'teacher'],
            5: ['gender', 'women', 'equality', 'female', 'empowerment'],
            6: ['water', 'sanitation', 'hygiene', 'clean', 'wastewater'],
            7: ['energy', 'renewable', 'solar', 'electricity', 'power'],
            8: ['employment', 'work', 'economic', 'growth', 'job'],
            9: ['industry', 'innovation', 'infrastructure', 'technology'],
            10: ['inequality', 'equality', 'inclusion', 'discrimination'],
            11: ['cities', 'urban', 'sustainable', 'community', 'housing'],
            12: ['consumption', 'production', 'waste', 'sustainable'],
            13: ['climate', 'carbon', 'emission', 'warming', 'environmental'],
            14: ['ocean', 'marine', 'water', 'sea', 'aquatic'],
            15: ['forest', 'biodiversity', 'land', 'ecosystem', 'wildlife'],
            16: ['peace', 'justice', 'institutions', 'governance', 'rights'],
            17: ['partnership', 'collaboration', 'cooperation', 'global']
        }
        
        text_lower = text.lower()
        keywords = sdg_keywords.get(sdg_number, [])
        matched = [kw for kw in keywords if kw in text_lower]
        
        return matched[:5]
    
    def _fallback_prediction(self, text: str, top_k: int = 3):
        """Fallback rule-based prediction"""
        print("🔄 Using fallback prediction")
        
        text_lower = text.lower()
        
        sdg_scores = {}
        for sdg_num in range(1, 18):
            keywords = self._extract_keywords(text, sdg_num)
            score = len(keywords) * 0.15
            if score > 0:
                sdg_scores[sdg_num] = score
        
        sorted_sdgs = sorted(sdg_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for sdg_num, score in sorted_sdgs:
            results.append({
                'sdg_number': sdg_num,
                'sdg_name': self.sdg_labels[sdg_num - 1],
                'confidence': min(0.8, score),
                'matched_keywords': self._extract_keywords(text, sdg_num),
                'explanation': f'Rule-based match. Confidence: {score:.2f}'
            })
        
        if not results:
            results = [{
                'sdg_number': 0,
                'sdg_name': 'No Clear Match',
                'confidence': 0.0,
                'matched_keywords': [],
                'explanation': 'No SDGs detected with sufficient confidence'
            }]
        
        return results
