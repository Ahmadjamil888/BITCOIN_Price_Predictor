import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime, timedelta
import requests
from newsapi import NewsApiClient
import ta

class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.sequence_length = 60
        
    def load_historical_data(self, csv_path):
        """Load and process historical Bitcoin data"""
        print(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Original columns: {list(df.columns)}")
        print(f"Data shape: {df.shape}")
        
        # Handle timestamp conversion
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        elif 'Date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Date'])
        else:
            # Create timestamp from index if no timestamp column
            df['timestamp'] = pd.date_range(start='2009-01-01', periods=len(df), freq='D')
        
        df = df.sort_values('timestamp')
        
        # Ensure we have a price column
        if 'Close' not in df.columns and 'output_amount' in df.columns:
            df['Close'] = df['output_amount']
        elif 'Close' not in df.columns and 'price' in df.columns:
            df['Close'] = df['price']
        elif 'Close' not in df.columns:
            # Use the first numeric column as price
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df['Close'] = df[numeric_cols[0]]
            else:
                raise ValueError("No suitable price column found in dataset")
        
        print(f"Processed data shape: {df.shape}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df
    
    def get_live_price_data(self, period='1y'):
        """Fetch live Bitcoin price data"""
        btc = yf.Ticker("BTC-USD")
        data = btc.history(period=period)
        data.reset_index(inplace=True)
        return data
    
    def add_technical_indicators(self, df):
        """Add technical analysis indicators"""
        df_copy = df.copy()
        
        # Find price column
        price_col = None
        for col in ['Close', 'close', 'output_amount', 'price']:
            if col in df_copy.columns:
                price_col = col
                break
        
        if price_col is None:
            print("Warning: No price column found, using first numeric column")
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                price_col = numeric_cols[0]
            else:
                return df_copy
        
        print(f"Using {price_col} as price column")
        
        try:
            # Ensure price column is numeric and handle missing values
            df_copy[price_col] = pd.to_numeric(df_copy[price_col], errors='coerce')
            df_copy[price_col] = df_copy[price_col].ffill().bfill()
            
            # RSI
            if len(df_copy) >= 14:
                df_copy['RSI'] = ta.momentum.RSIIndicator(df_copy[price_col], window=14).rsi()
            else:
                df_copy['RSI'] = 50.0  # Neutral RSI for short sequences
            
            # MACD
            if len(df_copy) >= 26:
                macd = ta.trend.MACD(df_copy[price_col])
                df_copy['MACD'] = macd.macd()
                df_copy['MACD_signal'] = macd.macd_signal()
            else:
                df_copy['MACD'] = 0.0
                df_copy['MACD_signal'] = 0.0
            
            # Bollinger Bands
            if len(df_copy) >= 20:
                bb = ta.volatility.BollingerBands(df_copy[price_col], window=20)
                df_copy['BB_upper'] = bb.bollinger_hband()
                df_copy['BB_lower'] = bb.bollinger_lband()
                df_copy['BB_middle'] = bb.bollinger_mavg()
            else:
                df_copy['BB_upper'] = df_copy[price_col] * 1.02
                df_copy['BB_lower'] = df_copy[price_col] * 0.98
                df_copy['BB_middle'] = df_copy[price_col]
            
            # Moving averages
            df_copy['MA_7'] = df_copy[price_col].rolling(window=min(7, len(df_copy))).mean()
            df_copy['MA_21'] = df_copy[price_col].rolling(window=min(21, len(df_copy))).mean()
            df_copy['MA_50'] = df_copy[price_col].rolling(window=min(50, len(df_copy))).mean()
            
            # Volume-based indicators if volume exists
            volume_cols = ['Volume', 'volume', 'tx_count']
            volume_col = None
            for col in volume_cols:
                if col in df_copy.columns:
                    volume_col = col
                    break
            
            if volume_col:
                df_copy['Volume_MA'] = df_copy[volume_col].rolling(window=min(20, len(df_copy))).mean()
            
        except Exception as e:
            print(f"Error adding technical indicators: {e}")
            # Add default values if calculation fails
            for col in ['RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'BB_middle', 'MA_7', 'MA_21', 'MA_50']:
                if col not in df_copy.columns:
                    df_copy[col] = 0.0
        
        return df_copy
    
    def get_news_sentiment(self, api_key):
        """Fetch Bitcoin news and analyze sentiment"""
        try:
            newsapi = NewsApiClient(api_key=api_key)
            articles = newsapi.get_everything(
                q='bitcoin OR cryptocurrency OR BTC',
                language='en',
                sort_by='publishedAt',
                page_size=50
            )
            
            # Simple sentiment scoring (can be enhanced with proper NLP)
            sentiment_score = 0
            for article in articles['articles']:
                title = article['title'].lower()
                description = article['description'] or ""
                description = description.lower()
                
                positive_words = ['rise', 'bull', 'gain', 'profit', 'surge', 'rally', 'up']
                negative_words = ['fall', 'bear', 'loss', 'crash', 'drop', 'decline', 'down']
                
                for word in positive_words:
                    if word in title or word in description:
                        sentiment_score += 1
                        
                for word in negative_words:
                    if word in title or word in description:
                        sentiment_score -= 1
            
            return sentiment_score / len(articles['articles']) if articles['articles'] else 0
            
        except Exception as e:
            print(f"Error fetching news: {e}")
            return 0
    
    def prepare_sequences(self, data, target_col='Close'):
        """Prepare sequences for LSTM training"""
        print(f"Preparing sequences with target column: {target_col}")
        
        # Find target column
        if target_col not in data.columns:
            for col in ['Close', 'close', 'output_amount', 'price']:
                if col in data.columns:
                    target_col = col
                    break
        
        if target_col not in data.columns:
            raise ValueError(f"Target column {target_col} not found in data")
        
        # Select features (exclude non-numeric and identifier columns)
        exclude_cols = ['timestamp', 'Date', 'date', 'height']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Ensure all feature columns are numeric
        features = data[feature_cols].copy()
        for col in feature_cols:
            features[col] = pd.to_numeric(features[col], errors='coerce')
        
        # Fill missing values
        features = features.ffill().bfill().fillna(0)
        
        print(f"Feature columns: {feature_cols}")
        print(f"Features shape: {features.shape}")
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Find target column index
        target_idx = feature_cols.index(target_col)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_features)):
            X.append(scaled_features[i-self.sequence_length:i])
            y.append(scaled_features[i, target_idx])
            
        print(f"Created {len(X)} sequences of length {self.sequence_length}")
        return np.array(X), np.array(y)