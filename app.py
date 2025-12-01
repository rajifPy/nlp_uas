import os
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from dotenv import load_dotenv
from utils.pdf_extractor import PDFExtractor
from utils.model_loader import ModelLoader
import tempfile
import requests

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default_secret_key_for_development')

# Initialize Supabase client (optional)
supabase = None
try:
    from supabase import create_client, Client
    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_key = os.environ.get('SUPABASE_KEY')
    
    if supabase_url and supabase_key:
        supabase = create_client(supabase_url, supabase_key)
        print("✅ Supabase connected")
except Exception as e:
    print(f"⚠️ Supabase disabled: {e}")

# Initialize components
pdf_extractor = PDFExtractor()
model_loader = ModelLoader(os.getenv('MODEL_PATH', './models'))

# Hugging Face fallback
HF_API_TOKEN = os.getenv('HF_API_TOKEN')
HF_API_URL = os.getenv('HF_API_URL', 'https://api-inference.huggingface.co/models/facebook/bart-large-mnli')

SDG_LABELS = [
    'No Poverty - ending poverty in all its forms',
    'Zero Hunger - ending hunger and promoting sustainable agriculture',
    'Good Health and Well-being - ensuring healthy lives',
    'Quality Education - ensuring inclusive education',
    'Gender Equality - achieving gender equality',
    'Clean Water and Sanitation - ensuring water availability',
    'Affordable and Clean Energy - ensuring clean energy access',
    'Decent Work and Economic Growth - promoting economic growth',
    'Industry Innovation and Infrastructure - building infrastructure',
    'Reduced Inequality - reducing inequality',
    'Sustainable Cities and Communities - making cities sustainable',
    'Responsible Consumption and Production - ensuring sustainable consumption',
    'Climate Action - combating climate change',
    'Life Below Water - conserving oceans',
    'Life on Land - protecting terrestrial ecosystems',
    'Peace Justice and Strong Institutions - promoting peace',
    'Partnerships for the Goals - strengthening partnerships'
]

def predict_with_hf_api(text: str, top_k: int = 3):
    """Predict using Hugging Face API"""
    if not HF_API_TOKEN:
        return None
    
    try:
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        payload = {
            "inputs": text[:500],
            "parameters": {
                "candidate_labels": SDG_LABELS,
                "multi_label": True
            }
        }
        
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            labels = data.get('labels', [])
            scores = data.get('scores', [])
            
            results = []
            for i in range(min(top_k, len(labels))):
                sdg_name = labels[i].split(' - ')[0]
                results.append({
                    'sdg_number': i + 1,
                    'sdg_name': sdg_name,
                    'confidence': float(scores[i]),
                    'matched_keywords': model_loader._extract_keywords(text, i + 1),
                    'explanation': f"AI analysis: {sdg_name} ({scores[i]:.1%} confidence)"
                })
            
            return results
        
    except Exception as e:
        print(f"HF API error: {e}")
    
    return None

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle PDF upload"""
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if not file.filename.lower().endswith('.pdf'):
        flash('Only PDF files are supported', 'error')
        return redirect(url_for('index'))
    
    try:
        # Save temporarily
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)
        
        # Extract content
        content = pdf_extractor.extract_content(file_path)
        analysis_text = f"{content['title']} {content['abstract']} {' '.join(content['keywords'])} {content['full_text']}"
        
        # Try HF API first, then local model
        model_results = predict_with_hf_api(analysis_text, top_k=3)
        
        if not model_results:
            # Fallback to local model
            model_results = model_loader.predict_sdgs(analysis_text, top_k=3)
        
        # Save to database if available
        extraction_id = str(uuid.uuid4())
        
        if supabase:
            try:
                user_id = session.get('user_id', 'anonymous_user')
                supabase.table('extractions').insert({
                    'id': extraction_id,
                    'user_id': user_id,
                    'document_name': file.filename,
                    'title': content['title'],
                    'abstract': content['abstract'],
                    'keywords': content['keywords'],
                    'sdg_results': model_results,
                    'created_at': datetime.now().isoformat()
                }).execute()
            except Exception as e:
                print(f"Database error: {e}")
        
        # Cleanup
        os.remove(file_path)
        os.rmdir(temp_dir)
        
        # Store in session
        session['extraction_results'] = {
            'metadata': pdf_extractor.extract_metadata(file_path) if os.path.exists(file_path) else {},
            'content': content,
            'model_results': model_results,
            'extraction_id': extraction_id,
            'document_name': file.filename
        }
        
        return redirect(url_for('results'))
        
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/results')
def results():
    """Display results"""
    if 'extraction_results' not in session:
        flash('No results found', 'error')
        return redirect(url_for('index'))
    
    return render_template('results.html', **session['extraction_results'])

@app.route('/history')
def history():
    """View history"""
    extractions = []
    
    if supabase:
        try:
            user_id = session.get('user_id', 'anonymous_user')
            response = supabase.table('extractions')\
                .select('*')\
                .eq('user_id', user_id)\
                .order('created_at', desc=True)\
                .limit(10)\
                .execute()
            
            if response.data:
                extractions = response.data
        except Exception as e:
            print(f"Database error: {e}")
    
    return render_template('history.html', extractions=extractions)

@app.route('/api/extract', methods=['POST'])
def api_extract():
    """API endpoint"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        # Save temporarily
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)
        
        # Extract
        content = pdf_extractor.extract_content(file_path)
        analysis_text = f"{content['title']} {content['abstract']} {' '.join(content['keywords'])} {content['full_text']}"
        
        # Predict
        results = predict_with_hf_api(analysis_text, top_k=3)
        if not results:
            results = model_loader.predict_sdgs(analysis_text, top_k=3)
        
        # Cleanup
        os.remove(file_path)
        os.rmdir(temp_dir)
        
        return jsonify({
            'success': True,
            'document': {
                'title': content['title'],
                'abstract': content['abstract'],
                'keywords': content['keywords']
            },
            'sdg_analysis': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loader.sdg_model is not None,
        'hf_api': HF_API_TOKEN is not None,
        'database': supabase is not None
    })

# For Railway/Render
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
