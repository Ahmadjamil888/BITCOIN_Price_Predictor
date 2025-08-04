from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys

app = Flask(__name__)
CORS(app)

# Simple LSTM model (same as in training)
class SimpleBitcoinLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(SimpleBitcoinLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Global variables
model = None
scaler = None
feature_cols = None

def load_simple_model():
    """Load the simple trained model"""
    global model, scaler, feature_cols
    
    try:
        # Load scaler
        scaler = joblib.load('saved_models/simple_scaler.pkl')
        
        # Load model
        checkpoint = torch.load('saved_models/simple_bitcoin_model.pth', map_location='cpu')
        input_size = checkpoint['input_size']
        feature_cols = checkpoint['feature_cols']
        
        model = SimpleBitcoinLSTM(input_size, hidden_size=32, num_layers=1)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("Simple model loaded successfully!")
        print(f"Features: {feature_cols}")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def get_current_btc_price():
    """Get current Bitcoin price"""
    try:
        btc = yf.Ticker("BTC-USD")
        data = btc.history(period="1d")
        return float(data['Close'].iloc[-1])
    except:
        return 50000.0  # Fallback price

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None,
        "model_type": "simple_lstm"
    })

@app.route('/api/current-price', methods=['GET'])
def get_current_price():
    """Get current Bitcoin price"""
    try:
        price = get_current_btc_price()
        return jsonify({
            "price": price,
            "timestamp": datetime.now().isoformat(),
            "currency": "USD"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/historical-data', methods=['GET'])
def get_historical_data():
    """Get historical Bitcoin price data"""
    try:
        period = request.args.get('period', '3mo')
        btc = yf.Ticker("BTC-USD")
        data = btc.history(period=period)
        
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
    """Predict Bitcoin price using the simple model"""
    global model, scaler, feature_cols
    
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Get recent data
        btc = yf.Ticker("BTC-USD")
        recent_data = btc.history(period="2mo")
        
        if len(recent_data) < 30:
            return jsonify({"error": "Insufficient recent data"}), 400
        
        # Prepare features similar to training
        df = recent_data.copy()
        df['price'] = df['Close']
        df['price_change'] = df['price'].pct_change()
        df['price_ma_5'] = df['price'].rolling(window=5).mean()
        df['price_ma_10'] = df['price'].rolling(window=10).mean()
        df['volatility'] = df['price'].rolling(window=10).std()
        
        # Select only the features used in training
        available_features = []
        for col in feature_cols:
            if col in df.columns:
                available_features.append(col)
            else:
                # Add dummy feature if missing
                df[col] = 0
                available_features.append(col)
        
        # Clean and scale data
        feature_data = df[feature_cols].ffill().bfill().fillna(0)
        scaled_data = scaler.transform(feature_data)
        
        # Get last 30 days for prediction
        last_sequence = scaled_data[-30:]
        
        # Make predictions
        steps = request.json.get('steps', 7) if request.json else 7
        predictions = []
        current_price = float(recent_data['Close'].iloc[-1])
        
        # Simple prediction approach
        model.eval()
        with torch.no_grad():
            for step in range(steps):
                # Predict next value
                X = torch.FloatTensor(last_sequence).unsqueeze(0)
                pred_scaled = model(X).item()
                
                # Inverse transform to get actual price
                dummy_array = np.zeros((1, len(feature_cols)))
                dummy_array[0, 0] = pred_scaled
                pred_price = scaler.inverse_transform(dummy_array)[0, 0]
                
                # Generate trading signal
                price_change = (pred_price - current_price) / current_price
                
                if price_change > 0.02:
                    signal = "BUY"
                    reason = "AI predicts upward trend"
                elif price_change < -0.02:
                    signal = "SELL"
                    reason = "AI predicts downward trend"
                else:
                    signal = "HOLD"
                    reason = "AI predicts stable price"
                
                predictions.append({
                    "day": step + 1,
                    "predicted_price": float(pred_price),
                    "signal": signal,
                    "confidence": min(80.0, 60.0 + abs(price_change) * 100),
                    "reason": reason
                })
                
                # Update sequence for next prediction
                new_row = last_sequence[-1].copy()
                new_row[0] = pred_scaled
                last_sequence = np.vstack([last_sequence[1:], new_row])
        
        return jsonify({
            "current_price": current_price,
            "predictions": predictions,
            "timestamp": datetime.now().isoformat(),
            "model_type": "simple_lstm"
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/technical-indicators', methods=['GET'])
def get_technical_indicators():
    """Get technical indicators"""
    try:
        btc = yf.Ticker("BTC-USD")
        data = btc.history(period="1mo")
        
        # Calculate simple indicators
        latest_price = float(data['Close'].iloc[-1])
        
        # Simple moving averages
        ma_7 = float(data['Close'].rolling(window=7).mean().iloc[-1])
        ma_21 = float(data['Close'].rolling(window=21).mean().iloc[-1])
        ma_50 = float(data['Close'].rolling(window=min(50, len(data))).mean().iloc[-1])
        
        # Simple RSI calculation
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        indicators = {
            "rsi": float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0,
            "macd": 0.0,  # Simplified
            "macd_signal": 0.0,  # Simplified
            "bb_upper": latest_price * 1.02,
            "bb_lower": latest_price * 0.98,
            "ma_7": ma_7,
            "ma_21": ma_21,
            "ma_50": ma_50,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(indicators)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/news-sentiment', methods=['GET'])
def get_news_sentiment():
    """Get mock news sentiment"""
    try:
        # Generate realistic sentiment based on recent price movement
        btc = yf.Ticker("BTC-USD")
        recent = btc.history(period="7d")
        
        if len(recent) > 1:
            price_change = (recent['Close'].iloc[-1] - recent['Close'].iloc[0]) / recent['Close'].iloc[0]
            sentiment_score = np.tanh(price_change * 5)  # Scale and bound between -1 and 1
        else:
            sentiment_score = 0.0
        
        return jsonify({
            "sentiment_score": float(sentiment_score),
            "sentiment_label": "Positive" if sentiment_score > 0.1 else "Negative" if sentiment_score < -0.1 else "Neutral",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Loading simple model...")
    if load_simple_model():
        print("Starting Flask server on http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please train the model first with:")
        print("python train_simple.py")