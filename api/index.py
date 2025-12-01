from flask import Flask, request, jsonify, render_template_string
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Global variables for lazy loading
supabase_client = None
supabase_queries = None
pdf_extractor = None
model_loader = None

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
    """Lazy initialize PDF extractor and model"""
    global pdf_extractor, model_loader
    
    if pdf_extractor is not None:
        return pdf_extractor, model_loader
    
    try:
        from utils.pdf_extractor import PDFExtractor
        from utils.model_loader import ModelLoader
        
        pdf_extractor = PDFExtractor()
        model_loader = ModelLoader(os.getenv('MODEL_PATH', '/tmp/models'))
        
        print("✅ Extractors initialized")
    except Exception as e:
        print(f"⚠️ Extractor init error: {e}")
        traceback.print_exc()
    
    return pdf_extractor, model_loader

# HTML Template
HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SDGs Extractor</title>
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
            max-width: 800px;
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
        .status {
            background: #f0f4f8;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .status-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #e0e0e0;
        }
        .status-ok { color: #22c55e; }
        .status-error { color: #ef4444; }
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
        }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
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
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌱 SDGs Extractor</h1>
        
        <div class="status">
            <h3>System Status</h3>
            <div class="status-item">
                <span>Model Loaded:</span>
                <span class="{{ 'status-ok' if model_loaded else 'status-error' }}">
                    {{ '✅ Ready' if model_loaded else '❌ Not Available' }}
                </span>
            </div>
            <div class="status-item">
                <span>Database:</span>
                <span class="{{ 'status-ok' if db_connected else 'status-error' }}">
                    {{ '✅ Connected' if db_connected else '⚠️ Disabled' }}
                </span>
            </div>
        </div>
        
        {% if not model_loaded %}
        <div class="error-box">
            <h3>⚠️ Model Not Available</h3>
            <p>Please set the following environment variables in Vercel:</p>
            <ul>
                <li><code>MODEL_URL</code> - Direct download URL to your model file</li>
                <li><code>HF_MODEL_URL</code> - Optional: Hugging Face model URL</li>
            </ul>
            <p style="margin-top: 15px;">
                <strong>Example:</strong><br>
                <code>MODEL_URL=https://huggingface.co/your-username/your-model/resolve/main/model.joblib</code>
            </p>
        </div>
        {% endif %}
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <div style="font-size: 4em; margin-bottom: 20px;">📄</div>
                <h3>Click to Upload PDF</h3>
                <input type="file" id="fileInput" name="file" accept=".pdf" required style="display: none;">
            </div>
            <div id="fileName"></div>
            <center>
                <button type="submit" class="btn" id="submitBtn" {{ 'disabled' if not model_loaded else '' }}>
                    Analyze Document
                </button>
            </center>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing your document...</p>
        </div>
        
        <div id="results"></div>
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
                fileName.style.color = '#667eea';
                fileName.style.fontWeight = 'bold';
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
            let html = '<h2>📊 Analysis Results</h2>';
            html += `<h3>${data.document.title}</h3>`;
            
            const sdgs = data.sdg_analysis || [];
            sdgs.forEach(sdg => {
                const confidence = (sdg.confidence * 100).toFixed(1);
                html += `
                    <div style="background: #f8f9ff; padding: 20px; margin: 15px 0; border-radius: 10px; border-left: 5px solid #667eea;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3>SDG ${sdg.sdg_number}: ${sdg.sdg_name}</h3>
                            <span style="background: #667eea; color: white; padding: 5px 15px; border-radius: 20px;">${confidence}%</span>
                        </div>
                        <p style="margin: 10px 0;">${sdg.explanation}</p>
                        <div>
                            <strong>Keywords:</strong><br>
                            ${sdg.matched_keywords.map(kw => `<span style="background: #e0e7ff; color: #667eea; padding: 3px 10px; border-radius: 12px; margin: 3px; display: inline-block;">${kw}</span>`).join('')}
                        </div>
                    </div>
                `;
            });
            
            html += '<center><button class="btn" onclick="location.reload()">Analyze Another</button></center>';
            
            results.innerHTML = html;
            results.style.display = 'block';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Home page with system status"""
    try:
        # Lazy init
        init_extractors()
        
        model_loaded = model_loader is not None and model_loader.sdg_model is not None
        db_connected = init_supabase() is not None
        
        return render_template_string(
            HOME_TEMPLATE,
            model_loaded=model_loaded,
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
        # Initialize components
        extractor, loader = init_extractors()
        
        if not extractor or not loader:
            return jsonify({
                'success': False,
                'error': 'System not initialized. Please check server logs.'
            }), 500
        
        if not loader.sdg_model:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please set MODEL_URL environment variable.'
            }), 503
        
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
            
            # Run model prediction
            results = loader.predict_sdgs(analysis_text, top_k=3)
            
            return jsonify({
                'success': True,
                'document': {
                    'title': content['title'],
                    'abstract': content['abstract'][:200] + '...',
                    'keywords': content['keywords']
                },
                'sdg_analysis': results
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
        extractor, loader = init_extractors()
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': loader is not None and loader.sdg_model is not None,
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
