from flask import Flask, request, jsonify, render_template_string
import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)

# Auto-download model jika belum ada
print("🔍 Checking for model...")
try:
    from api.download_model import download_model
    download_model()
except Exception as e:
    print(f"⚠️ Error in auto-download: {e}")

# Import utilities
try:
    from utils.pdf_extractor import PDFExtractor
    from utils.model_loader import ModelLoader
    pdf_extractor = PDFExtractor()
    model_loader = ModelLoader(os.getenv('MODEL_PATH', '/tmp/models'))
except Exception as e:
    print(f"Error importing utilities: {e}")
    pdf_extractor = None
    model_loader = None

# HTML Templates (inline untuk Vercel)
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
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 40px;
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 60px 20px;
            text-align: center;
            background: #f8f9ff;
            transition: all 0.3s;
            cursor: pointer;
        }
        .upload-area:hover {
            background: #eef1ff;
            border-color: #764ba2;
        }
        .upload-icon {
            font-size: 4em;
            margin-bottom: 20px;
        }
        input[type="file"] { display: none; }
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
        .btn:hover { transform: scale(1.05); }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        #fileName {
            margin-top: 20px;
            color: #667eea;
            font-weight: bold;
        }
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
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
        .results {
            display: none;
            margin-top: 30px;
        }
        .sdg-card {
            background: #f8f9ff;
            border-left: 5px solid #667eea;
            padding: 20px;
            margin: 15px 0;
            border-radius: 10px;
        }
        .sdg-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .confidence {
            background: #667eea;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }
        .keywords {
            margin-top: 10px;
        }
        .keyword-tag {
            background: #e0e7ff;
            color: #667eea;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.9em;
            margin: 3px;
            display: inline-block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌱 SDGs Extractor</h1>
        <p class="subtitle">AI-powered Sustainable Development Goals Analysis</p>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📄</div>
                <h3>Click to Upload PDF</h3>
                <p>Or drag and drop your file here</p>
                <input type="file" id="fileInput" name="file" accept=".pdf" required>
            </div>
            <div id="fileName"></div>
            <center>
                <button type="submit" class="btn" id="submitBtn">Analyze Document</button>
            </center>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing your document...</p>
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
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Error uploading file: ' + error.message);
            } finally {
                loading.style.display = 'none';
                submitBtn.disabled = false;
            }
        });

        function displayResults(data) {
            const sdgs = data.sdg_analysis;
            let html = '<h2>📊 Analysis Results</h2>';
            html += `<h3>📄 ${data.document.title}</h3>`;
            html += '<div style="margin: 20px 0;">';
            
            sdgs.forEach((sdg, index) => {
                const confidence = (sdg.confidence * 100).toFixed(1);
                html += `
                    <div class="sdg-card">
                        <div class="sdg-header">
                            <h3>🎯 SDG ${sdg.sdg_number}: ${sdg.sdg_name}</h3>
                            <span class="confidence">${confidence}%</span>
                        </div>
                        <p>${sdg.explanation}</p>
                        <div class="keywords">
                            <strong>Keywords:</strong><br>
                            ${sdg.matched_keywords.map(kw => `<span class="keyword-tag">${kw}</span>`).join('')}
                        </div>
                    </div>
                `;
            });
            
            html += '</div>';
            html += '<center><button class="btn" onclick="location.reload()">Analyze Another Document</button></center>';
            
            results.innerHTML = html;
            results.style.display = 'block';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HOME_TEMPLATE)

@app.route('/api/extract', methods=['POST'])
def extract():
    """API endpoint for SDG extraction"""
    try:
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
            content = pdf_extractor.extract_content(tmp_path)
            analysis_text = f"{content['title']} {content['abstract']} {' '.join(content['keywords'])} {content['full_text']}"
            
            # Run model prediction
            results = model_loader.predict_sdgs(analysis_text, top_k=3)
            
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
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loader.sdg_model is not None if model_loader else False
    })

# Vercel handler
def handler(request):
    with app.request_context(request.environ):
        return app.full_dispatch_request()