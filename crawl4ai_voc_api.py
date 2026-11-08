from flask import Flask, request, jsonify, render_template_string
from crawl4ai import AsyncWebCrawler
from openai import OpenAI
import asyncio
import os
from typing import List, Dict, Optional
import json
import sqlite3
from datetime import datetime
from contextlib import contextmanager

app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Database setup
DB_NAME = 'voc_data.db'

def init_db():
    """Initialize SQLite database with tables."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Scrape results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scrape_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            markdown TEXT,
            cleaned_html TEXT,
            success BOOLEAN,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Analysis results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scrape_id INTEGER,
            analysis_type TEXT,
            analysis_data TEXT,
            model TEXT,
            tokens_used INTEGER,
            success BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (scrape_id) REFERENCES scrape_results (id)
        )
    ''')
    
    # Projects table for organizing scrapes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Project scrapes junction table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS project_scrapes (
            project_id INTEGER,
            scrape_id INTEGER,
            FOREIGN KEY (project_id) REFERENCES projects (id),
            FOREIGN KEY (scrape_id) REFERENCES scrape_results (id),
            PRIMARY KEY (project_id, scrape_id)
        )
    ''')
    
    conn.commit()
    conn.close()

@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

class VoCAnalyzer:
    """Voice of Customer analyzer using Crawl4AI and OpenAI."""
    
    def __init__(self):
        self.crawler = None
    
    async def scrape_url(self, url: str, css_selector: Optional[str] = None) -> Dict:
        """Scrape a single URL using Crawl4AI."""
        async with AsyncWebCrawler(verbose=True) as crawler:
            result = await crawler.arun(
                url=url,
                css_selector=css_selector,
                bypass_cache=True
            )
            
            scrape_data = {
                'url': url,
                'markdown': result.markdown,
                'cleaned_html': result.cleaned_html,
                'success': result.success,
                'metadata': json.dumps(result.metadata) if result.metadata else None
            }
            
            # Save to database
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO scrape_results (url, markdown, cleaned_html, success, metadata)
                    VALUES (?, ?, ?, ?, ?)
                ''', (url, result.markdown, result.cleaned_html, result.success, scrape_data['metadata']))
                conn.commit()
                scrape_data['id'] = cursor.lastrowid
            
            return scrape_data
    
    async def scrape_multiple_urls(self, urls: List[str], css_selector: Optional[str] = None) -> List[Dict]:
        """Scrape multiple URLs concurrently."""
        tasks = [self.scrape_url(url, css_selector) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'url': urls[i],
                    'success': False,
                    'error': str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def analyze_with_gpt(self, content: str, analysis_type: str = "general", scrape_id: Optional[int] = None) -> Dict:
        """Analyze scraped content using OpenAI GPT."""
        
        prompts = {
            "general": """Analyze this customer feedback/review content and provide Voice of Customer insights:

Content:
{content}

Please provide:
1. Key themes and sentiments
2. Common pain points
3. Feature requests or desires
4. Overall sentiment (positive/negative/neutral)
5. Actionable recommendations

Format your response as JSON with these keys: themes, pain_points, feature_requests, sentiment, recommendations""",
            
            "sentiment": """Perform sentiment analysis on this customer content:

Content:
{content}

Provide a detailed sentiment breakdown with:
1. Overall sentiment score (-1 to 1)
2. Positive aspects mentioned
3. Negative aspects mentioned
4. Neutral observations
5. Emotional tone

Format as JSON with keys: sentiment_score, positive_aspects, negative_aspects, neutral_aspects, emotional_tone""",
            
            "competitive": """Analyze this content for competitive intelligence and market positioning:

Content:
{content}

Identify:
1. Competitor mentions
2. Product comparisons
3. Market gaps or opportunities
4. Customer expectations vs reality
5. Differentiation opportunities

Format as JSON with keys: competitors, comparisons, market_gaps, expectations, differentiation"""
        }
        
        prompt = prompts.get(analysis_type, prompts["general"]).format(content=content[:15000])
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a Voice of Customer analyst expert. Analyze customer feedback and provide structured insights in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            result = {
                'success': True,
                'analysis': analysis,
                'model': response.model,
                'tokens_used': response.usage.total_tokens
            }
            
            # Save to database
            if scrape_id:
                with get_db() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO analysis_results (scrape_id, analysis_type, analysis_data, model, tokens_used, success)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (scrape_id, analysis_type, json.dumps(analysis), response.model, response.usage.total_tokens, True))
                    conn.commit()
                    result['analysis_id'] = cursor.lastrowid
            
            return result
        
        except Exception as e:
            # Save error to database
            if scrape_id:
                with get_db() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO analysis_results (scrape_id, analysis_type, analysis_data, success)
                        VALUES (?, ?, ?, ?)
                    ''', (scrape_id, analysis_type, json.dumps({'error': str(e)}), False))
                    conn.commit()
            
            return {
                'success': False,
                'error': str(e)
            }

voc_analyzer = VoCAnalyzer()

# Frontend HTML
FRONTEND_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VoC Analysis Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #ff8c42 0%, #e8532e 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .card {
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
            border-bottom: 2px solid #e0e0e0;
        }
        
        .tab {
            padding: 12px 24px;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 16px;
            color: #666;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
        }
        
        .tab:hover {
            color: #ff8c42;
        }
        
        .tab.active {
            color: #ff8c42;
            border-bottom-color: #ff8c42;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        input[type="text"],
        textarea,
        select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        
        input[type="text"]:focus,
        textarea:focus,
        select:focus {
            outline: none;
            border-color: #ff8c42;
        }
        
        textarea {
            min-height: 100px;
            resize: vertical;
        }
        
        .url-list {
            margin-bottom: 10px;
        }
        
        .url-item {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }
        
        .url-item input {
            flex: 1;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #ff8c42 0%, #e8532e 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(255, 140, 66, 0.4);
        }
        
        .btn-secondary {
            background: #f0f0f0;
            color: #333;
        }
        
        .btn-secondary:hover {
            background: #e0e0e0;
        }
        
        .btn-small {
            padding: 8px 16px;
            font-size: 14px;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #ff8c42;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .results {
            margin-top: 30px;
        }
        
        .result-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        
        .result-card h3 {
            color: #ff8c42;
            margin-bottom: 15px;
        }
        
        .result-section {
            margin-bottom: 15px;
        }
        
        .result-section h4 {
            color: #333;
            margin-bottom: 8px;
            font-size: 16px;
        }
        
        .result-list {
            list-style: none;
            padding-left: 0;
        }
        
        .result-list li {
            padding: 8px 0;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .result-list li:last-child {
            border-bottom: none;
        }
        
        .sentiment-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
        }
        
        .sentiment-positive {
            background: #d4edda;
            color: #155724;
        }
        
        .sentiment-negative {
            background: #f8d7da;
            color: #721c24;
        }
        
        .sentiment-neutral {
            background: #fff3cd;
            color: #856404;
        }
        
        .history-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        .history-table th,
        .history-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .history-table th {
            background: #f8f9fa;
            font-weight: 600;
            color: #333;
        }
        
        .history-table tr:hover {
            background: #f8f9fa;
        }
        
        .view-btn {
            color: #ff8c42;
            cursor: pointer;
            text-decoration: underline;
        }
        
        .view-btn:hover {
            color: #e8532e;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Voice of Customer Analysis</h1>
            <p>Scrape customer feedback and get AI-powered insights</p>
        </div>
        
        <div class="card">
            <div class="tabs">
                <button class="tab active" onclick="switchTab('scrape')">Scrape & Analyze</button>
                <button class="tab" onclick="switchTab('batch')">Batch Analysis</button>
                <button class="tab" onclick="switchTab('history')">History</button>
            </div>
            
            <!-- Scrape & Analyze Tab -->
            <div id="scrape-tab" class="tab-content active">
                <form id="scrapeForm">
                    <div class="form-group">
                        <label>URL to Scrape</label>
                        <input type="text" id="url" placeholder="https://example.com/reviews" required>
                    </div>
                    
                    <div class="form-group">
                        <label>CSS Selector (Optional)</label>
                        <input type="text" id="cssSelector" placeholder=".review-content">
                        <small style="color: #666;">Target specific elements on the page</small>
                    </div>
                    
                    <div class="form-group">
                        <label>Analysis Type</label>
                        <select id="analysisType">
                            <option value="general">General VoC Insights</option>
                            <option value="sentiment">Sentiment Analysis</option>
                            <option value="competitive">Competitive Intelligence</option>
                        </select>
                    </div>
                    
                    <button type="submit" class="btn btn-primary">Analyze</button>
                </form>
                
                <div id="loading1" class="loading">
                    <div class="spinner"></div>
                    <p>Scraping and analyzing...</p>
                </div>
                
                <div id="results1" class="results"></div>
            </div>
            
            <!-- Batch Analysis Tab -->
            <div id="batch-tab" class="tab-content">
                <form id="batchForm">
                    <div class="form-group">
                        <label>URLs to Scrape</label>
                        <div id="urlList" class="url-list">
                            <div class="url-item">
                                <input type="text" placeholder="https://example.com/reviews/page1" required>
                                <button type="button" class="btn btn-secondary btn-small" onclick="removeUrl(this)">Remove</button>
                            </div>
                        </div>
                        <button type="button" class="btn btn-secondary btn-small" onclick="addUrl()">+ Add URL</button>
                    </div>
                    
                    <div class="form-group">
                        <label>CSS Selector (Optional)</label>
                        <input type="text" id="batchCssSelector" placeholder=".review-content">
                    </div>
                    
                    <button type="submit" class="btn btn-primary">Batch Analyze</button>
                </form>
                
                <div id="loading2" class="loading">
                    <div class="spinner"></div>
                    <p>Processing multiple sources...</p>
                </div>
                
                <div id="results2" class="results"></div>
            </div>
            
            <!-- History Tab -->
            <div id="history-tab" class="tab-content">
                <button class="btn btn-secondary" onclick="loadHistory()">Refresh History</button>
                <div id="historyResults"></div>
            </div>
        </div>
    </div>
    
    <script>
        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            event.target.classList.add('active');
            document.getElementById(tabName + '-tab').classList.add('active');
            
            if (tabName === 'history') {
                loadHistory();
            }
        }
        
        function addUrl() {
            const urlList = document.getElementById('urlList');
            const newItem = document.createElement('div');
            newItem.className = 'url-item';
            newItem.innerHTML = `
                <input type="text" placeholder="https://example.com/reviews/page" required>
                <button type="button" class="btn btn-secondary btn-small" onclick="removeUrl(this)">Remove</button>
            `;
            urlList.appendChild(newItem);
        }
        
        function removeUrl(btn) {
            const urlList = document.getElementById('urlList');
            if (urlList.children.length > 1) {
                btn.parentElement.remove();
            }
        }
        
        document.getElementById('scrapeForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const url = document.getElementById('url').value;
            const cssSelector = document.getElementById('cssSelector').value;
            const analysisType = document.getElementById('analysisType').value;
            
            document.getElementById('loading1').style.display = 'block';
            document.getElementById('results1').innerHTML = '';
            
            try {
                const response = await fetch('/scrape-and-analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        url: url,
                        css_selector: cssSelector || null,
                        analysis_type: analysisType
                    })
                });
                
                const data = await response.json();
                displayResults(data, 'results1');
            } catch (error) {
                document.getElementById('results1').innerHTML = `<div class="result-card"><p style="color: red;">Error: ${error.message}</p></div>`;
            } finally {
                document.getElementById('loading1').style.display = 'none';
            }
        });
        
        document.getElementById('batchForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const urlInputs = document.querySelectorAll('#urlList input');
            const urls = Array.from(urlInputs).map(input => input.value).filter(url => url);
            const cssSelector = document.getElementById('batchCssSelector').value;
            
            document.getElementById('loading2').style.display = 'block';
            document.getElementById('results2').innerHTML = '';
            
            try {
                const response = await fetch('/batch-analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        urls: urls,
                        css_selector: cssSelector || null
                    })
                });
                
                const data = await response.json();
                displayBatchResults(data, 'results2');
            } catch (error) {
                document.getElementById('results2').innerHTML = `<div class="result-card"><p style="color: red;">Error: ${error.message}</p></div>`;
            } finally {
                document.getElementById('loading2').style.display = 'none';
            }
        });
        
        function displayResults(data, targetId) {
            const resultsDiv = document.getElementById(targetId);
            
            if (data.error) {
                resultsDiv.innerHTML = `<div class="result-card"><p style="color: red;">Error: ${data.error}</p></div>`;
                return;
            }
            
            const result = data.results[0];
            if (!result.scrape_success) {
                resultsDiv.innerHTML = `<div class="result-card"><p style="color: red;">Scrape failed: ${result.error}</p></div>`;
                return;
            }
            
            const analysis = result.analysis.analysis;
            let html = '<div class="result-card">';
            html += `<h3>Analysis Results for: ${result.url}</h3>`;
            
            // Display based on what's in the analysis
            for (const [key, value] of Object.entries(analysis)) {
                html += '<div class="result-section">';
                html += `<h4>${formatKey(key)}</h4>`;
                
                if (Array.isArray(value)) {
                    html += '<ul class="result-list">';
                    value.forEach(item => {
                        html += `<li>${typeof item === 'object' ? JSON.stringify(item) : item}</li>`;
                    });
                    html += '</ul>';
                } else if (typeof value === 'object') {
                    html += '<ul class="result-list">';
                    for (const [k, v] of Object.entries(value)) {
                        html += `<li><strong>${formatKey(k)}:</strong> ${v}</li>`;
                    }
                    html += '</ul>';
                } else if (key === 'sentiment' || key === 'overall_sentiment') {
                    const sentimentClass = value.toLowerCase().includes('positive') ? 'positive' : 
                                          value.toLowerCase().includes('negative') ? 'negative' : 'neutral';
                    html += `<span class="sentiment-badge sentiment-${sentimentClass}">${value}</span>`;
                } else {
                    html += `<p>${value}</p>`;
                }
                
                html += '</div>';
            }
            
            html += '</div>';
            resultsDiv.innerHTML = html;
        }
        
        function displayBatchResults(data, targetId) {
            const resultsDiv = document.getElementById(targetId);
            
            if (data.error) {
                resultsDiv.innerHTML = `<div class="result-card"><p style="color: red;">Error: ${data.error}</p></div>`;
                return;
            }
            
            const analysis = data.aggregate_analysis.analysis;
            let html = '<div class="result-card">';
            html += `<h3>Aggregate Analysis (${data.sources_analyzed} sources)</h3>`;
            
            for (const [key, value] of Object.entries(analysis)) {
                html += '<div class="result-section">';
                html += `<h4>${formatKey(key)}</h4>`;
                
                if (Array.isArray(value)) {
                    html += '<ul class="result-list">';
                    value.forEach(item => {
                        html += `<li>${typeof item === 'object' ? JSON.stringify(item) : item}</li>`;
                    });
                    html += '</ul>';
                } else if (typeof value === 'object') {
                    html += '<ul class="result-list">';
                    for (const [k, v] of Object.entries(value)) {
                        html += `<li><strong>${formatKey(k)}:</strong> ${v}</li>`;
                    }
                    html += '</ul>';
                } else {
                    html += `<p>${value}</p>`;
                }
                
                html += '</div>';
            }
            
            html += '</div>';
            resultsDiv.innerHTML = html;
        }
        
        async function loadHistory() {
            const historyDiv = document.getElementById('historyResults');
            historyDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>Loading history...</p></div>';
            
            try {
                const response = await fetch('/history');
                const data = await response.json();
                
                let html = '<table class="history-table">';
                html += '<thead><tr><th>Date</th><th>URL</th><th>Status</th><th>Analysis</th></tr></thead>';
                html += '<tbody>';
                
                data.forEach(item => {
                    const date = new Date(item.created_at).toLocaleString();
                    const status = item.success ? '✅ Success' : '❌ Failed';
                    html += `<tr>
                        <td>${date}</td>
                        <td>${item.url}</td>
                        <td>${status}</td>
                        <td><span class="view-btn" onclick="viewDetails(${item.id})">View Details</span></td>
                    </tr>`;
                });
                
                html += '</tbody></table>';
                historyDiv.innerHTML = html;
            } catch (error) {
                historyDiv.innerHTML = `<p style="color: red;">Error loading history: ${error.message}</p>`;
            }
        }
        
        async function viewDetails(scrapeId) {
            try {
                const response = await fetch(`/scrape/${scrapeId}`);
                const data = await response.json();
                
                alert(`URL: ${data.url}\\n\\nContent Preview:\\n${data.markdown.substring(0, 500)}...\\n\\nAnalyses: ${data.analyses.length}`);
            } catch (error) {
                alert('Error loading details: ' + error.message);
            }
        }
        
        function formatKey(key) {
            return key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Serve the frontend."""
    return render_template_string(FRONTEND_HTML)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'Crawl4AI VoC API'})

@app.route('/scrape', methods=['POST'])
def scrape():
    """Scrape one or more URLs."""
    data = request.json
    
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    urls = data.get('urls', [])
    if isinstance(data.get('url'), str):
        urls = [data['url']]
    
    if not urls:
        return jsonify({'error': 'No URLs provided'}), 400
    
    css_selector = data.get('css_selector')
    
    try:
        if len(urls) == 1:
            result = asyncio.run(voc_analyzer.scrape_url(urls[0], css_selector))
        else:
            result = asyncio.run(voc_analyzer.scrape_multiple_urls(urls, css_selector))
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze pre-scraped content with GPT."""
    data = request.json
    
    if not data or 'content' not in data:
        return jsonify({'error': 'No content provided'}), 400
    
    content = data['content']
    analysis_type = data.get('analysis_type', 'general')
    scrape_id = data.get('scrape_id')
    
    if analysis_type not in ['general', 'sentiment', 'competitive']:
        return jsonify({'error': 'Invalid analysis_type. Choose: general, sentiment, or competitive'}), 400
    
    result = voc_analyzer.analyze_with_gpt(content, analysis_type, scrape_id)
    return jsonify(result)

@app.route('/scrape-and-analyze', methods=['POST'])
def scrape_and_analyze():
    """Scrape URL(s) and immediately analyze with GPT."""
    data = request.json
    
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    urls = data.get('urls', [])
    if isinstance(data.get('url'), str):
        urls = [data['url']]
    
    if not urls:
        return jsonify({'error': 'No URLs provided'}), 400
    
    css_selector = data.get('css_selector')
    analysis_type = data.get('analysis_type', 'general')
    
    try:
        # Scrape
        if len(urls) == 1:
            scrape_result = asyncio.run(voc_analyzer.scrape_url(urls[0], css_selector))
            scrape_results = [scrape_result]
        else:
            scrape_results = asyncio.run(voc_analyzer.scrape_multiple_urls(urls, css_selector))
        
        # Analyze each result
        analyses = []
        for scrape_result in scrape_results:
            if scrape_result.get('success'):
                content = scrape_result.get('markdown', '')
                scrape_id = scrape_result.get('id')
                analysis = voc_analyzer.analyze_with_gpt(content, analysis_type, scrape_id)
                analyses.append({
                    'url': scrape_result['url'],
                    'scrape_success': True,
                    'analysis': analysis
                })
            else:
                analyses.append({
                    'url': scrape_result.get('url'),
                    'scrape_success': False,
                    'error': scrape_result.get('error', 'Unknown error')
                })
        
        return jsonify({
            'results': analyses,
            'total_urls': len(urls),
            'successful_scrapes': sum(1 for a in analyses if a.get('scrape_success'))
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """Scrape multiple URLs and provide aggregate VoC insights."""
    data = request.json
    
    if not data or 'urls' not in data:
        return jsonify({'error': 'No URLs provided'}), 400
    
    urls = data['urls']
    css_selector = data.get('css_selector')
    
    try:
        # Scrape all URLs
        scrape_results = asyncio.run(voc_analyzer.scrape_multiple_urls(urls, css_selector))
        
        # Combine all content
        combined_content = "\n\n---\n\n".join([
            f"Source: {r['url']}\n{r.get('markdown', '')}"
            for r in scrape_results if r.get('success')
        ])
        
        if not combined_content:
            return jsonify({'error': 'No content successfully scraped'}), 400
        
        # Analyze combined content
        analysis = voc_analyzer.analyze_with_gpt(combined_content, "general")
        
        return jsonify({
            'aggregate_analysis': analysis,
            'sources_analyzed': len([r for r in scrape_results if r.get('success')]),
            'total_urls': len(urls),
            'scrape_details': scrape_results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Get scraping history."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, url, success, created_at
                FROM scrape_results
                ORDER BY created_at DESC
                LIMIT 50
            ''')
            results = cursor.fetchall()
            
            history = []
            for row in results:
                history.append({
                    'id': row['id'],
                    'url': row['url'],
                    'success': bool(row['success']),
                    'created_at': row['created_at']
                })
            
            return jsonify(history)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/scrape/<int:scrape_id>', methods=['GET'])
def get_scrape_details(scrape_id):
    """Get details of a specific scrape."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get scrape data
            cursor.execute('''
                SELECT * FROM scrape_results WHERE id = ?
            ''', (scrape_id,))
            scrape = cursor.fetchone()
            
            if not scrape:
                return jsonify({'error': 'Scrape not found'}), 404
            
            # Get associated analyses
            cursor.execute('''
                SELECT * FROM analysis_results WHERE scrape_id = ?
            ''', (scrape_id,))
            analyses = cursor.fetchall()
            
            result = {
                'id': scrape['id'],
                'url': scrape['url'],
                'markdown': scrape['markdown'],
                'success': bool(scrape['success']),
                'created_at': scrape['created_at'],
                'analyses': []
            }
            
            for analysis in analyses:
                result['analyses'].append({
                    'id': analysis['id'],
                    'analysis_type': analysis['analysis_type'],
                    'analysis_data': json.loads(analysis['analysis_data']) if analysis['analysis_data'] else None,
                    'model': analysis['model'],
                    'tokens_used': analysis['tokens_used'],
                    'success': bool(analysis['success']),
                    'created_at': analysis['created_at']
                })
            
            return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/projects', methods=['GET', 'POST'])
def manage_projects():
    """Create or list projects."""
    if request.method == 'GET':
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM projects ORDER BY created_at DESC')
                projects = cursor.fetchall()
                
                result = []
                for project in projects:
                    result.append({
                        'id': project['id'],
                        'name': project['name'],
                        'description': project['description'],
                        'created_at': project['created_at']
                    })
                
                return jsonify(result)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    elif request.method == 'POST':
        data = request.json
        
        if not data or 'name' not in data:
            return jsonify({'error': 'Project name required'}), 400
        
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO projects (name, description)
                    VALUES (?, ?)
                ''', (data['name'], data.get('description', '')))
                conn.commit()
                
                return jsonify({
                    'id': cursor.lastrowid,
                    'name': data['name'],
                    'description': data.get('description', '')
                })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get statistics about scrapes and analyses."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Total scrapes
            cursor.execute('SELECT COUNT(*) as count FROM scrape_results')
            total_scrapes = cursor.fetchone()['count']
            
            # Successful scrapes
            cursor.execute('SELECT COUNT(*) as count FROM scrape_results WHERE success = 1')
            successful_scrapes = cursor.fetchone()['count']
            
            # Total analyses
            cursor.execute('SELECT COUNT(*) as count FROM analysis_results')
            total_analyses = cursor.fetchone()['count']
            
            # Total tokens used
            cursor.execute('SELECT SUM(tokens_used) as total FROM analysis_results')
            total_tokens = cursor.fetchone()['total'] or 0
            
            # Recent activity (last 7 days)
            cursor.execute('''
                SELECT COUNT(*) as count 
                FROM scrape_results 
                WHERE created_at >= datetime('now', '-7 days')
            ''')
            recent_scrapes = cursor.fetchone()['count']
            
            return jsonify({
                'total_scrapes': total_scrapes,
                'successful_scrapes': successful_scrapes,
                'success_rate': round(successful_scrapes / total_scrapes * 100, 2) if total_scrapes > 0 else 0,
                'total_analyses': total_analyses,
                'total_tokens_used': total_tokens,
                'recent_activity_7_days': recent_scrapes
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize database
    init_db()
    print("✅ Database initialized")
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  WARNING: OPENAI_API_KEY environment variable not set!")
    else:
        print("✅ OpenAI API key found")
    
    print("🚀 Starting VoC Analysis API...")
    print("📊 Dashboard: http://localhost:5008")
    print("📚 API Docs: Check the README for endpoint details")
    
    app.run(debug=True, host='0.0.0.0', port=5008)