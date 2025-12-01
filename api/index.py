from flask import Flask, request, jsonify, render_template_string
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime
import traceback
import requests
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Global variables for lazy loading
supabase_client = None
supabase_queries = None
pdf_extractor = None

# Hugging Face API Configuration
HF_API_URL = os.getenv('HF_API_URL', 'https://api-inference.huggingface.co/models/facebook/bart-large-mnli')
HF_API_TOKEN = os.getenv('HF_API_TOKEN')

# SDG Labels for classification
SDG_LABELS = [
    'No Poverty - ending poverty in all its forms',
    'Zero Hunger - ending hunger and promoting sustainable agriculture',
    'Good Health and Well-being - ensuring healthy lives and promoting well-being',
    'Quality Education - ensuring inclusive and equitable quality education',
    'Gender Equality - achieving gender equality and empowering women and girls',
    'Clean Water and Sanitation - ensuring availability of water and sanitation',
    'Affordable and Clean Energy - ensuring access to affordable and clean energy',
    'Decent Work and Economic Growth - promoting sustained economic growth and decent work',
    'Industry Innovation and Infrastructure - building resilient infrastructure and fostering innovation',
    'Reduced Inequality - reducing inequality within and among countries',
    'Sustainable Cities and Communities - making cities and human settlements sustainable',
    'Responsible Consumption and Production - ensuring sustainable consumption and production',
    'Climate Action - taking urgent action to combat climate change',
    'Life Below Water - conserving and sustainably using oceans and marine resources',
    'Life on Land - protecting and restoring terrestrial ecosystems and biodiversity',
    'Peace Justice and Strong Institutions - promoting peaceful and inclusive societies',
    'Partnerships for the Goals - strengthening global partnerships for sustainable development'
]

def init_supabase():
    """Lazy initialize Supabase"""
    global supabase_client, supabase_queries
    
    if supabase_client is not None:
        return supabase_client
    
    try:
        from supabase import create_client, Client
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if supabase_url and supabase_key:
            supabase_client = create_client(supabase_url, supabase_key)
            from utils.supabase_queries import SupabaseQueries
            supabase_queries = SupabaseQueries(supabase_client)
            print("✅ Supabase initialized")
        else:
            print("⚠️ Supabase credentials not found")
    except Exception as e:
        print(f"⚠️ Supabase init error: {e}")
    
    return supabase_client

def init_extractors():
    """Lazy initialize PDF extractor"""
    global pdf_extractor
    
    if pdf_extractor is not None:
        return pdf_extractor
    
    try:
        from utils.pdf_extractor import PDFExtractor
        pdf_extractor = PDFExtractor()
        print("✅ PDF Extractor initialized")
    except Exception as e:
        print(f"⚠️ Extractor init error: {e}")
        traceback.print_exc()
    
    return pdf_extractor

def predict_sdgs_with_hf(text: str, top_k: int = 3):
    """
    Predict SDGs using Hugging Face Inference API (Zero-shot Classification)
    """
    if not HF_API_TOKEN:
        print("⚠️ HF_API_TOKEN not set, using fallback")
        return fallback_prediction(text, top_k)
    
    try:
        # Truncate text to avoid token limits
        max_length = 500
        text_sample = text[:max_length] if len(text) > max_length else text
        
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        
        payload = {
            "inputs": text_sample,
            "parameters": {
                "candidate_labels": SDG_LABELS,
                "multi_label": True
            }
        }
        
        print(f"🔍 Calling Hugging Face API...")
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Parse results
            labels = data.get('labels', [])
            scores = data.get('scores', [])
            
            results = []
            for i in range(min(top_k, len(labels))):
                label = labels[i]
                score = scores[i]
                
                # Extract SDG number from label
                sdg_number = i + 1
                sdg_name = label.split(' - ')[0] if ' - ' in label else label
                
                # Extract keywords
                keywords = extract_keywords_from_text(text, sdg_number)
                
                results.append({
                    'sdg_number': sdg_number,
                    'sdg_name': sdg_name,
                    'confidence': float(score),
                    'matched_keywords': keywords,
                    'explanation': f"AI-predicted alignment with {sdg_name}. Confidence: {score:.2%}",
                    'source': 'huggingface_inference_api'
                })
            
            print(f"✅ HF API prediction complete")
            return results
        
        elif response.status_code == 503:
            print("⚠️ Model is loading, using fallback")
            return fallback_prediction(text, top_k)
        
        else:
            print(f"❌ HF API error: {response.status_code}")
            return fallback_prediction(text, top_k)
            
    except requests.Timeout:
        print("⚠️ HF API timeout, using fallback")
        return fallback_prediction(text, top_k)
    
    except Exception as e:
        print(f"❌ HF API error: {e}")
        traceback.print_exc()
        return fallback_prediction(text, top_k)

def extract_keywords_from_text(text: str, sdg_number: int) -> list:
    """Extract relevant keywords for specific SDG"""
    sdg_keywords = {
        1: ['poverty', 'poor', 'income', 'economic', 'financial', 'disadvantage'],
        2: ['hunger', 'food', 'nutrition', 'agriculture', 'farming', 'malnutrition'],
        3: ['health', 'medical', 'disease', 'healthcare', 'wellbeing', 'wellness'],
        4: ['education', 'school', 'learning', 'student', 'teacher', 'literacy'],
        5: ['gender', 'women', 'equality', 'female', 'empowerment', 'rights'],
        6: ['water', 'sanitation', 'hygiene', 'clean', 'wastewater', 'sewage'],
        7: ['energy', 'renewable', 'solar', 'electricity', 'power', 'wind'],
        8: ['employment', 'work', 'economic', 'growth', 'job', 'labor'],
        9: ['industry', 'innovation', 'infrastructure', 'technology', 'research'],
        10: ['inequality', 'equality', 'inclusion', 'discrimination', 'disparity'],
        11: ['cities', 'urban', 'sustainable', 'community', 'housing', 'settlement'],
        12: ['consumption', 'production', 'waste', 'sustainable', 'recycling'],
        13: ['climate', 'carbon', 'emission', 'warming', 'environmental', 'greenhouse'],
        14: ['ocean', 'marine', 'water', 'sea', 'aquatic', 'fish'],
        15: ['forest', 'biodiversity', 'land', 'ecosystem', 'wildlife', 'conservation'],
        16: ['peace', 'justice', 'institutions', 'governance', 'rights', 'law'],
        17: ['partnership', 'collaboration', 'cooperation', 'global', 'alliance']
    }
    
    text_lower = text.lower()
    keywords = sdg_keywords.get(sdg_number, [])
    matched = [kw for kw in keywords if kw in text_lower]
    
    return matched[:5]

def fallback_prediction(text: str, top_k: int = 3):
    """Rule-based fallback prediction"""
    print("🔄 Using rule-based fallback")
    
    text_lower = text.lower()
    sdg_scores = {}
    
    for sdg_num in range(1, 18):
        keywords = extract_keywords_from_text(text, sdg_num)
        score = len(keywords) * 0.15
        if score > 0:
            sdg_scores[sdg_num] = min(0.85, score)
    
    sorted_sdgs = sorted(sdg_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    results = []
    for sdg_num, score in sorted_sdgs:
        sdg_name = SDG_LABELS[sdg_num - 1].split(' - ')[0]
        results.append({
            'sdg_number': sdg_num,
            'sdg_name': sdg_name,
            'confidence': score,
            'matched_keywords': extract_keywords_from_text(text, sdg_num),
            'explanation': f'Rule-based match. Found {len(extract_keywords_from_text(text, sdg_num))} relevant keywords',
            'source': 'rule_based_fallback'
        })
    
    if not results:
        results = [{
            'sdg_number': 0,
            'sdg_name': 'No Clear Match',
            'confidence': 0.0,
            'matched_keywords': [],
            'explanation': 'No SDGs detected with sufficient confidence',
            'source': 'fallback'
        }]
    
    return results

# HTML Template
HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SDGs Extractor - AI Powered</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 { 
            color: #667eea;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .status {
            background: linear-gradient(135deg, #f0f4f8 0%, #e0e7ff 100%);
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 5px solid #667eea;
        }
        .status h3 {
            color: #667eea;
            margin-bottom: 15px;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .status-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .status-ok { color: #22c55e; font-weight: bold; }
        .status-warn { color: #f59e0b; font-weight: bold; }
        .status-error { color: #ef4444; font-weight: bold; }
        .badge {
            background: #667eea;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.85em;
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 60px 20px;
            text-align: center;
            background: #f8f9ff;
            transition: all 0.3s;
            cursor: pointer;
            margin: 30px 0;
        }
        .upload-area:hover {
            background: #eef1ff;
            border-color: #764ba2;
            transform: translateY(-2px);
        }
        .upload-icon {
            font-size: 4em;
            margin-bottom: 20px;
            animation: bounce 2s infinite;
        }
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 25px;
            font-size: 1.1em;
            cursor: pointer;
            margin-top: 20px;
            transition: transform 0.2s;
        }
        .btn:hover {
            transform: scale(1.05);
        }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .info-box {
            background: #fff3cd;
            border: 2px solid #ffc107;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .error-box {
            background: #fee;
            border: 2px solid #f88;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .results {
            display: none;
            margin-top: 30px;
        }
        .sdg-card {
            background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
            border-left: 5px solid #667eea;
            padding: 25px;
            margin: 20px 0;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .sdg-card:hover {
            transform: translateX(5px);
        }
        .sdg-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .confidence {
            background: #667eea;
            color: white;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 1.1em;
        }
        .keywords {
            margin-top: 15px;
        }
        .keyword-tag {
            background: #e0e7ff;
            color: #667eea;
            padding: 5px 12px;
            border-radius: 12px;
            font-size: 0.9em;
            margin: 3px;
            display: inline-block;
        }
        .source-badge {
            background: #22c55e;
            color: white;
            padding: 3px 10px;
            border-radius: 10px;
            font-size: 0.8em;
            margin-left: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌱 SDGs Extractor</h1>
        <p class="subtitle">AI-powered Sustainable Development Goals Analysis</p>
        
        <div class="status">
            <h3>🔧 System Status</h3>
            <div class="status-grid">
                <div class="status-item">
                    <span>🤖 AI Model:</span>
                    <span class="{{ 'status-ok' if hf_configured else 'status-warn' }}">
                        {{ 'HF API' if hf_configured else 'Fallback' }}
                    </span>
                </div>
                <div class="status-item">
                    <span>💾 Database:</span>
                    <span class="{{ 'status-ok' if db_connected else 'status-warn' }}">
                        {{ 'Connected' if db_connected else 'Offline' }}
                    </span>
                </div>
                <div class="status-item">
                    <span>📄 PDF Parser:</span>
                    <span class="status-ok">Active</span>
                </div>
            </div>
        </div>
        
        {% if not hf_configured %}
        <div class="info-box">
            <h3>ℹ️ Using Rule-Based Fallback</h3>
            <p>For better accuracy, set <code>HF_API_TOKEN</code> in Vercel environment variables.</p>
            <p style="margin-top: 10px;">Get your free token at: <a href="https://huggingface.co/settings/tokens" target="_blank">https://huggingface.co/settings/tokens</a></p>
        </div>
        {% endif %}
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📄</div>
                <h3>Click to Upload PDF Document</h3>
                <p>Or drag and drop your file here</p>
                <input type="file" id="fileInput" name="file" accept=".pdf" required style="display: none;">
            </div>
            <div id="fileName" style="text-align: center; margin-top: 10px; color: #667eea; font-weight: bold;"></div>
            <center>
                <button type="submit" class="btn" id="submitBtn">
                    🔍 Analyze Document
                </button>
            </center>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <h3>🧠 Analyzing your document with AI...</h3>
            <p>This may take 10-30 seconds</p>
        </div>
        
        <div class="results" id="results"></div>
    </div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const fileName = document.getElementById('fileName');
        const uploadForm = document.getElementById('uploadForm');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const submitBtn = document.getElementById('submitBtn');

        fileInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                fileName.textContent = '📎 ' + this.files[0].name;
            }
        });

        uploadForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            if (!fileInput.files.length) {
                alert('Please select a PDF file');
                return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            loading.style.display = 'block';
            results.style.display = 'none';
            submitBtn.disabled = true;

            try {
                const response = await fetch('/api/extract', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.success) {
                    displayResults(data);
                } else {
                    alert('Error: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                loading.style.display = 'none';
                submitBtn.disabled = false;
            }
        });

        function displayResults(data) {
            const sdgs = data.sdg_analysis || [];
            let html = '<h2>📊 Analysis Results</h2>';
            html += `<h3 style="color: #667eea; margin: 15px 0;">${data.document.title}</h3>`;
            
            if (data.document.abstract) {
                html += `<p style="color: #666; margin-bottom: 20px;"><strong>Abstract:</strong> ${data.document.abstract}</p>`;
            }
            
            sdgs.forEach((sdg, index) => {
                const confidence = (sdg.confidence * 100).toFixed(1);
                const sourceColor = sdg.source === 'huggingface_inference_api' ? '#22c55e' : 
                                   sdg.source === 'rule_based_fallback' ? '#f59e0b' : '#667eea';
                
                html += `
                    <div class="sdg-card">
                        <div class="sdg-header">
                            <div>
                                <h3 style="color: #667eea; margin: 0;">
                                    🎯 SDG ${sdg.sdg_number}: ${sdg.sdg_name}
                                    <span class="source-badge" style="background: ${sourceColor};">
                                        ${sdg.source === 'huggingface_inference_api' ? '🤖 AI' : 
                                          sdg.source === 'rule_based_fallback' ? '📋 Rule' : '🔍 Auto'}
                                    </span>
                                </h3>
                            </div>
                            <span class="confidence">${confidence}%</span>
                        </div>
                        <p style="margin: 10px 0; color: #555;">${sdg.explanation}</p>
                        ${sdg.matched_keywords && sdg.matched_keywords.length > 0 ? `
                        <div class="keywords">
                            <strong style="color: #667eea;">🔑 Keywords:</strong><br>
                            ${sdg.matched_keywords.map(kw => 
                                `<span class="keyword-tag">${kw}</span>`
                            ).join('')}
                        </div>
                        ` : ''}
                    </div>
                `;
            });
            
            html += '<center><button class="btn" onclick="location.reload()">📄 Analyze Another Document</button></center>';
            
            results.innerHTML = html;
            results.style.display = 'block';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Home page"""
    try:
        init_extractors()
        
        hf_configured = HF_API_TOKEN is not None and len(HF_API_TOKEN) > 0
        db_connected = init_supabase() is not None
        
        return render_template_string(
            HOME_TEMPLATE,
            hf_configured=hf_configured,
            db_connected=db_connected
        )
    except Exception as e:
        print(f"Error in index: {e}")
        traceback.print_exc()
        return f"Error: {str(e)}", 500

@app.route('/api/extract', methods=['POST'])
def extract():
    """API endpoint for SDG extraction"""
    try:
        extractor = init_extractors()
        
        if not extractor:
            return jsonify({
                'success': False,
                'error': 'PDF extractor not initialized'
            }), 500
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'success': False, 'error': 'Only PDF files supported'}), 400
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Extract content
            content = extractor.extract_content(tmp_path)
            analysis_text = f"{content['title']} {content['abstract']} {' '.join(content['keywords'])} {content['full_text']}"
            
            # Predict using HF API or fallback
            results = predict_sdgs_with_hf(analysis_text, top_k=3)
            
            return jsonify({
                'success': True,
                'document': {
                    'title': content['title'],
                    'abstract': content['abstract'][:200] + '...' if len(content['abstract']) > 200 else content['abstract'],
                    'keywords': content['keywords']
                },
                'sdg_analysis': results,
                'method': 'huggingface_api' if HF_API_TOKEN else 'rule_based'
            })
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except Exception as e:
        print(f"Error in extract: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health')
def health():
    """Health check endpoint"""
    try:
        extractor = init_extractors()
        
        return jsonify({
            'status': 'healthy',
            'hf_api_configured': HF_API_TOKEN is not None,
            'pdf_extractor': extractor is not None,
            'database_connected': init_supabase() is not None,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

# Vercel handler
def handler(request):
    with app.request_context(request.environ):
        return app.full_dispatch_request()
