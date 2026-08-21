"""
Flask REST API for Fake News Detection System.
Exposes endpoints for prediction, explainable AI, model metrics comparison,
dataset exploration, custom sample addition with live retraining, and batch processing.
"""

import os
import io
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from dataset import load_initial_dataset, get_dataset_stats
from ml_engine import FakeNewsMLEngine

app = Flask(__name__, static_folder='../frontend', static_url_path='')
app.config['SECRET_KEY'] = 'fake_news_detector_secret_key'
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Global State
dataset_df = load_initial_dataset()
ml_engine = FakeNewsMLEngine()

# Train models initially
print("Initializing Machine Learning Models on Benchmark Dataset...")
ml_engine.train(dataset_df)
print("ML Models successfully trained and ready!")

@app.route('/api/status', methods=['GET'])
def get_status():
    stats = get_dataset_stats(dataset_df)
    return jsonify({
        "status": "online",
        "is_trained": ml_engine.is_trained,
        "active_models": list(ml_engine.model_names.keys()),
        "dataset_stats": stats
    })

@app.route('/api/predict', methods=['POST'])
def predict_news():
    data = request.get_json() or {}
    text = data.get('text', '').strip()
    title = data.get('title', '').strip()
    model_key = data.get('model', 'ensemble')

    if not text and not title:
        return jsonify({"error": "Please provide article text or headline."}), 400

    full_input = f"{title} {text}".strip()
    
    try:
        res = ml_engine.predict(full_input, model_key=model_key)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/models/compare', methods=['GET'])
def compare_models():
    return jsonify({
        "models_metrics": ml_engine.metrics
    })

@app.route('/api/dataset', methods=['GET'])
def get_dataset():
    stats = get_dataset_stats(dataset_df)
    
    # Return samples
    samples = dataset_df.to_dict(orient='records')
    # Clean output dicts
    cleaned_samples = []
    for s in samples:
        cleaned_samples.append({
            "title": s.get("title", ""),
            "text": s.get("text", "")[:180] + "..." if len(s.get("text", "")) > 180 else s.get("text", ""),
            "label": int(s.get("label", 1)),
            "label_text": "REAL" if s.get("label", 1) == 1 else "FAKE"
        })
        
    return jsonify({
        "stats": stats,
        "samples": cleaned_samples
    })

@app.route('/api/dataset/add', methods=['POST'])
def add_dataset_sample():
    global dataset_df
    data = request.get_json() or {}
    title = data.get('title', '').strip()
    text = data.get('text', '').strip()
    label = data.get('label')

    if not text or label is None:
        return jsonify({"error": "Title/Text and Label (0 for Fake, 1 for Real) are required."}), 400

    try:
        label_int = int(label)
        if label_int not in [0, 1]:
            return jsonify({"error": "Label must be 0 (Fake) or 1 (Real)."}), 400

        new_row = pd.DataFrame([{
            "title": title if title else "User Added Sample",
            "text": text,
            "label": label_int,
            "full_text": f"{title} {text}".strip()
        }])
        
        dataset_df = pd.concat([dataset_df, new_row], ignore_index=True)
        
        # Retrain models with updated dataset
        updated_metrics = ml_engine.train(dataset_df)
        stats = get_dataset_stats(dataset_df)

        return jsonify({
            "message": "Sample added and models retrained successfully!",
            "updated_stats": stats,
            "updated_metrics": updated_metrics
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/batch', methods=['POST'])
def process_batch():
    """Processes array of texts or uploaded file."""
    articles = []
    
    if 'file' in request.files:
        uploaded_file = request.files['file']
        filename = uploaded_file.filename.lower()
        if filename.endswith('.csv'):
            df_upload = pd.read_csv(uploaded_file)
            for _, row in df_upload.iterrows():
                t = str(row.get('text', row.get('content', row.get('article', ''))))
                ttl = str(row.get('title', row.get('headline', '')))
                if t or ttl:
                    articles.append({"title": ttl, "text": t})
        elif filename.endswith('.txt'):
            lines = uploaded_file.read().decode('utf-8').split('\n')
            for line in lines:
                if line.strip():
                    articles.append({"title": "", "text": line.strip()})
    else:
        data = request.get_json() or {}
        articles = data.get('articles', [])

    if not articles:
        return jsonify({"error": "No valid articles found in batch request."}), 400

    results = []
    for item in articles[:50]: # limit to 50 for performance
        full_t = f"{item.get('title', '')} {item.get('text', '')}".strip()
        if full_t:
            pred = ml_engine.predict(full_t, model_key="ensemble")
            results.append({
                "title": item.get('title', 'Article'),
                "snippet": full_t[:120] + "..." if len(full_t) > 120 else full_t,
                "verdict": pred['verdict'],
                "credibility_score": pred['credibility_score'],
                "sensationalism_score": pred['sensationalism_score'],
                "objectivity_score": pred['objectivity_score']
            })

    return jsonify({
        "total_processed": len(results),
        "results": results
    })

# Serve Frontend static assets
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=True)
