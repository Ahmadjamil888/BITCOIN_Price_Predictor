from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
import joblib
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processor import DataProcessor
from models.bitcoin_predictor import BitcoinPredictor

app = Flask(__name__)
CORS(app)

# Global variables
model = None
processor = None
scaler = None

def load_model():
    """Load the trained model and scaler"""
    global model, processor, scaler
    
    try:
        # Load scaler
        scaler = joblib.load('../saved_models/scaler.pkl')
        
        # Initialize processor
        processor = DataProcessor()
        processor.scaler = scaler
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get input size from scaler
        input_size = scaler.n_features_in_
        model = BitcoinPredictor(input_size, device)
        model.load_model('../saved_models/bitcoin_model.pth')
        
        print("Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/api/current-price', methods=['GET'])
def get_current_price():
    """Get current Bitcoin price"""
    try:
        btc = yf.Ticker("BTC-USD")
        data = btc.history(period="1d")
        current_price = float(data['Close'].iloc[-1])
        
        return jsonify({
            "price": current_price,
            "timestamp": datetime.now().isoformat(),
            "currency": "USD"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/historical-data', methods=['GET'])
def get_historical_data():
    """Get historical Bitcoin price data"""
    try:
        period = request.args.get('period', '1y')
        btc = yf.Ticker("BTC-USD")
        data = btc.history(period=period)
        
        # Convert to JSON format
        historical_data = []
        for index, row in data.iterrows():
            historical_data.append({
                "date": index.strftime('%Y-%m-%d'),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": float(row['Volume'])
            })
        
        return jsonify(historical_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_price():
    """Predict Bitcoin price"""
    global model, processor
    
    if model is None or processor is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Get recent data for prediction
        btc = yf.Ticker("BTC-USD")
        recent_data = btc.history(period="3mo")
        
        # Add technical indicators
        recent_data = processor.add_technical_indicators(recent_data)
        recent_data = recent_data.dropna()
        
        if len(recent_data) < processor.sequence_length:
            return jsonify({"error": "Insufficient data for prediction"}), 400
        
        # Prepare last sequence
        feature_cols = [col for col in recent_data.columns if col not in ['Date']]
        features = recent_data[feature_cols].fillna(method='ffill').fillna(0)
        scaled_features = processor.scaler.transform(features)
        
        last_sequence = scaled_features[-processor.sequence_length:]
        
        # Make prediction
        steps = request.json.get('steps', 1) if request.json else 1
        predictions = model.predict_next_price(last_sequence, processor.scaler, steps)
        
        # Inverse transform predictions
        dummy_array = np.zeros((len(predictions), processor.scaler.n_features_in_))
        dummy_array[:, 0] = predictions  # Assuming price is first feature
        predictions_original = processor.scaler.inverse_transform(dummy_array)[:, 0]
        
        current_price = float(recent_data['Close'].iloc[-1])
        
        # Generate trading signals
        signals = []
        for i, pred_price in enumerate(predictions_original):
            signal, confidence, reason = model.get_trading_signal(current_price, pred_price)
            signals.append({
                "day": i + 1,
                "predicted_price": float(pred_price),
                "signal": signal,
                "confidence": float(confidence),
                "reason": reason
            })
        
        return jsonify({
            "current_price": current_price,
            "predictions": signals,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/news-sentiment', methods=['GET'])
def get_news_sentiment():
    """Get Bitcoin news sentiment"""
    try:
        # This would require a news API key
        # For demo purposes, return mock data
        sentiment_score = np.random.uniform(-1, 1)
        
        return jsonify({
            "sentiment_score": float(sentiment_score),
            "sentiment_label": "Positive" if sentiment_score > 0.1 else "Negative" if sentiment_score < -0.1 else "Neutral",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/technical-indicators', methods=['GET'])
def get_technical_indicators():
    """Get technical indicators for Bitcoin"""
    try:
        btc = yf.Ticker("BTC-USD")
        data = btc.history(period="1mo")
        
        # Add technical indicators
        if processor:
            data = processor.add_technical_indicators(data)
        
        latest = data.iloc[-1]
        
        indicators = {
            "rsi": float(latest.get('RSI', 0)),
            "macd": float(latest.get('MACD', 0)),
            "macd_signal": float(latest.get('MACD_signal', 0)),
            "bb_upper": float(latest.get('BB_upper', 0)),
            "bb_lower": float(latest.get('BB_lower', 0)),
            "ma_7": float(latest.get('MA_7', 0)),
            "ma_21": float(latest.get('MA_21', 0)),
            "ma_50": float(latest.get('MA_50', 0)),
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(indicators)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Loading model...")
    if load_model():
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please train the model first.")