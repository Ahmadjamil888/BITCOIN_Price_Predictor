import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
from data_processor import DataProcessor
from models.bitcoin_predictor import BitcoinPredictor
import joblib
import os

def train_bitcoin_model():
    """Train the Bitcoin price prediction model"""
    print("Loading and processing data...")
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Load historical data
    try:
        historical_data = processor.load_historical_data('../dataset.csv')
        print(f"Loaded {len(historical_data)} historical records")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure dataset.csv exists in the root directory")
        return None, None
    
    # Get live price data for recent trends (optional)
    try:
        live_data = processor.get_live_price_data('1y')
        print(f"Loaded {len(live_data)} live price records")
        
        # Try to combine datasets if they have compatible structure
        if not live_data.empty and len(historical_data) > 0:
            # Check if we can merge the data
            hist_max_date = historical_data['timestamp'].max()
            live_min_date = live_data['Date'].min()
            
            if hist_max_date < live_min_date:
                print("Combining historical and live data...")
                # Prepare live data to match historical format
                live_subset = live_data[['Date', 'Close']].copy()
                live_subset.columns = ['timestamp', 'Close']
                
                # Add missing columns with default values
                for col in historical_data.columns:
                    if col not in live_subset.columns and col != 'timestamp':
                        live_subset[col] = 0
                
                # Combine data
                combined_data = pd.concat([historical_data, live_subset], ignore_index=True)
                combined_data = combined_data.sort_values('timestamp')
                print(f"Combined dataset has {len(combined_data)} records")
            else:
                combined_data = historical_data
        else:
            combined_data = historical_data
            
    except Exception as e:
        print(f"Could not load live data: {e}")
        print("Using only historical data for training")
        combined_data = historical_data
    
    # Add technical indicators
    print("Adding technical indicators...")
    combined_data = processor.add_technical_indicators(combined_data)
    
    # Remove rows with NaN values (but keep some data)
    initial_len = len(combined_data)
    combined_data = combined_data.dropna()
    final_len = len(combined_data)
    print(f"Data after cleaning: {final_len} records (removed {initial_len - final_len} rows with NaN)")
    
    if final_len < processor.sequence_length * 2:
        print(f"Warning: Only {final_len} records available, which may be insufficient for training")
        print("Consider using a smaller sequence length or more data")
    
    # Prepare sequences for training
    print("Preparing training sequences...")
    X, y = processor.prepare_sequences(combined_data)
    print(f"Created {len(X)} training sequences")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Training set: {len(X_train)}")
    print(f"Validation set: {len(X_val)}")
    print(f"Test set: {len(X_test)}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    input_size = X.shape[2]
    model = BitcoinPredictor(input_size, device)
    
    # Train model
    print("Training model...")
    train_losses, val_losses = model.train(X_train, y_train, X_val, y_val, epochs=100)
    
    # Evaluate model
    print("Evaluating model...")
    test_predictions = model.predict(X_test)
    
    # Calculate metrics
    mse = np.mean((test_predictions.flatten() - y_test) ** 2)
    mae = np.mean(np.abs(test_predictions.flatten() - y_test))
    
    print(f"Test MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    
    # Save model and scaler
    os.makedirs('saved_models', exist_ok=True)
    model.save_model('saved_models/bitcoin_model.pth')
    joblib.dump(processor.scaler, 'saved_models/scaler.pkl')
    
    print("Model saved successfully!")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, test_predictions.flatten(), alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()
    
    return model, processor

if __name__ == "__main__":
    model, processor = train_bitcoin_model()