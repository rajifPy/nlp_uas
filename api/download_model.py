import os
import requests
from pathlib import Path

def download_model():
    """
    Download model saat cold start Vercel function.
    Model akan di-cache di /tmp/ untuk warm starts.
    """
    model_dir = Path('/tmp/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / 'BEST_MODEL_LightGBM_TFIDF.joblib'
    
    # Skip jika model sudah ada (warm start)
    if model_path.exists():
        file_size = model_path.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ Model already exists ({file_size:.2f} MB)")
        return True
    
    # Get download URL dari environment variable
    model_url = os.getenv('MODEL_URL')
    if not model_url:
        print("⚠️ MODEL_URL not set in environment variables")
        print("💡 Set MODEL_URL to your model's direct download link")
        return False
    
    try:
        print(f"⏬ Downloading model from: {model_url[:50]}...")
        
        # Download dengan streaming untuk file besar
        response = requests.get(model_url, timeout=60, stream=True)
        
        if response.status_code == 200:
            # Write file in chunks
            chunk_size = 8192
            downloaded = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Progress indicator setiap 10MB
                        if downloaded % (10 * 1024 * 1024) == 0:
                            print(f"📥 Downloaded: {downloaded / (1024*1024):.1f} MB")
            
            final_size = model_path.stat().st_size / (1024 * 1024)
            print(f"✅ Model downloaded successfully! Size: {final_size:.2f} MB")
            
            # Verify file bisa dibaca
            try:
                import joblib
                _ = joblib.load(model_path)
                print("✅ Model verification passed")
            except Exception as e:
                print(f"⚠️ Model verification failed: {str(e)}")
                os.remove(model_path)
                return False
            
            return True
        else:
            print(f"❌ Download failed with status code: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
    
    except requests.exceptions.Timeout:
        print("❌ Download timeout (60s exceeded)")
        return False
    
    except Exception as e:
        print(f"❌ Error downloading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def get_model_info():
    """Get info tentang model yang terdownload"""
    model_path = Path('/tmp/models/BEST_MODEL_LightGBM_TFIDF.joblib')
    
    if not model_path.exists():
        return {
            'exists': False,
            'message': 'Model not found. Please set MODEL_URL and restart.'
        }
    
    try:
        file_size = model_path.stat().st_size / (1024 * 1024)
        
        return {
            'exists': True,
            'path': str(model_path),
            'size_mb': round(file_size, 2),
            'message': f'Model loaded successfully ({file_size:.2f} MB)'
        }
    except Exception as e:
        return {
            'exists': False,
            'error': str(e),
            'message': 'Error reading model file'
        }

if __name__ == '__main__':
    # Test download
    success = download_model()
    if success:
        info = get_model_info()
        print(f"\n📊 Model Info: {info}")
    else:
        print("\n❌ Model download failed!")
