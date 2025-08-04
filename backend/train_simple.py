import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
import os

# Simple LSTM model for Bitcoin prediction
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

def prepare_bitcoin_data(csv_path, sequence_length=30, max_samples=50000):
    """Prepare Bitcoin data for training"""
    print(f"Loading data from {csv_path}")
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Handle timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    elif 'Date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['Date'])
    else:
        df['timestamp'] = pd.date_range(start='2009-01-01', periods=len(df), freq='D')
    
    df = df.sort_values('timestamp')
    
    # Handle Bitcoin blockchain data specifically
    if 'output_amount' in df.columns:
        # Convert satoshis to Bitcoin (divide by 100,000,000)
        df['price'] = pd.to_numeric(df['output_amount'], errors='coerce') / 100000000
        # Filter out zero values (early blocks with no transactions)
        df = df[df['price'] > 0]
        print(f"Filtered to {len(df)} blocks with transactions")
        
        # Sample data to reduce memory usage
        if len(df) > max_samples:
            print(f"Sampling {max_samples} records from {len(df)} for memory efficiency")
            # Take recent data and some historical samples
            recent_data = df.tail(max_samples // 2)
            historical_sample = df.head(len(df) - max_samples // 2).sample(n=max_samples // 2, random_state=42)
            df = pd.concat([historical_sample, recent_data]).sort_values('timestamp')
        
        # Use blockchain metrics as features
        feature_cols = ['price']
        
        # Add blockchain-specific features
        if 'tx_count' in df.columns:
            df['tx_count'] = pd.to_numeric(df['tx_count'], errors='coerce')
            feature_cols.append('tx_count')
            
        if 'difficulty' in df.columns:
            df['difficulty'] = pd.to_numeric(df['difficulty'], errors='coerce')
            # Log transform difficulty to handle large values
            df['difficulty'] = np.log1p(df['difficulty'])
            feature_cols.append('difficulty')
            
        if 'total_fees' in df.columns:
            df['total_fees'] = pd.to_numeric(df['total_fees'], errors='coerce') / 100000000  # Convert to BTC
            feature_cols.append('total_fees')
            
        if 'size' in df.columns:
            df['block_size'] = pd.to_numeric(df['size'], errors='coerce')
            feature_cols.append('block_size')
    else:
        # Handle regular price data
        price_col = None
        for col in ['Close', 'close', 'price']:
            if col in df.columns:
                price_col = col
                break
        
        if price_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                price_col = numeric_cols[0]
            else:
                raise ValueError("No numeric columns found for price data")
        
        df['price'] = pd.to_numeric(df[price_col], errors='coerce')
        df = df.dropna(subset=['price'])
        feature_cols = ['price']
    
    # Add technical indicators
    df['price_change'] = df['price'].pct_change()
    df['price_ma_5'] = df['price'].rolling(window=5).mean()
    df['price_ma_10'] = df['price'].rolling(window=10).mean()
    df['volatility'] = df['price'].rolling(window=10).std()
    
    feature_cols.extend(['price_change', 'price_ma_5', 'price_ma_10', 'volatility'])
    
    # Clean data
    df = df[feature_cols].ffill().bfill().fillna(0)
    
    print(f"Using features: {feature_cols}")
    print(f"Clean data shape: {df.shape}")
    print(f"Price range: {df['price'].min():.8f} to {df['price'].max():.8f} BTC")
    
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # Predict price (first column)
    
    X, y = np.array(X), np.array(y)
    print(f"Created {len(X)} sequences of length {sequence_length}")
    
    return X, y, scaler, feature_cols

def train_simple_model():
    """Train a simple Bitcoin prediction model"""
    print("=== Simple Bitcoin Price Predictor Training ===")
    
    # Prepare data with smaller sample size
    try:
        X, y, scaler, feature_cols = prepare_bitcoin_data('../dataset.csv', max_samples=10000)
    except Exception as e:
        print(f"Error preparing data: {e}")
        return False
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Initialize model
    input_size = X.shape[2]
    model = SimpleBitcoinLSTM(input_size, hidden_size=32, num_layers=1)  # Smaller model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Model input size: {input_size}")
    print("Starting training with batch processing...")
    
    # Training loop with batch processing
    train_losses = []
    val_losses = []
    batch_size = 32
    
    for epoch in range(50):  # Fewer epochs
        model.train()
        total_train_loss = 0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = torch.FloatTensor(X_train[i:i+batch_size])
            batch_y = torch.FloatTensor(y_train[i:i+batch_size])
            
            optimizer.zero_grad()
            train_pred = model(batch_X)
            train_loss = criterion(train_pred.squeeze(), batch_y)
            train_loss.backward()
            optimizer.step()
            
            total_train_loss += train_loss.item()
            num_batches += 1
        
        avg_train_loss = total_train_loss / num_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss_total = 0
            val_batches = 0
            for i in range(0, len(X_val), batch_size):
                batch_X_val = torch.FloatTensor(X_val[i:i+batch_size])
                batch_y_val = torch.FloatTensor(y_val[i:i+batch_size])
                
                val_pred = model(batch_X_val)
                val_loss = criterion(val_pred.squeeze(), batch_y_val)
                val_loss_total += val_loss.item()
                val_batches += 1
            
            avg_val_loss = val_loss_total / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/50, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    # Test the model
    model.eval()
    with torch.no_grad():
        test_loss_total = 0
        test_batches = 0
        test_predictions = []
        
        for i in range(0, len(X_test), batch_size):
            batch_X_test = torch.FloatTensor(X_test[i:i+batch_size])
            batch_y_test = torch.FloatTensor(y_test[i:i+batch_size])
            
            test_pred = model(batch_X_test)
            test_loss = criterion(test_pred.squeeze(), batch_y_test)
            test_loss_total += test_loss.item()
            test_batches += 1
            test_predictions.extend(test_pred.squeeze().numpy())
        
        avg_test_loss = test_loss_total / test_batches
    
    print(f'Final Test Loss: {avg_test_loss:.6f}')
    
    # Save model and scaler
    os.makedirs('saved_models', exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'feature_cols': feature_cols
    }, 'saved_models/simple_bitcoin_model.pth')
    
    joblib.dump(scaler, 'saved_models/simple_scaler.pkl')
    
    print("Model saved successfully!")
    
    # Plot results
    try:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        test_predictions = np.array(test_predictions)
        plt.scatter(y_test[:len(test_predictions)], test_predictions, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted')
        
        plt.tight_layout()
        plt.savefig('simple_training_results.png')
        print("Training plots saved as 'simple_training_results.png'")
    except Exception as e:
        print(f"Could not save plots: {e}")
    
    return True

if __name__ == "__main__":
    success = train_simple_model()
    if success:
        print("\nTraining completed successfully!")
        print("You can now start the API server with: python api/simple_app.py")
    else:
        print("\nTraining failed. Please check your dataset.")