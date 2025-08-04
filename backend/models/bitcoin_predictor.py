import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

class BitcoinLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.2):
        super(BitcoinLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # LSTM layers
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take the last output
        out = attn_out[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out

class BitcoinPredictor:
    def __init__(self, input_size, device='cpu'):
        self.device = device
        self.model = BitcoinLSTM(input_size).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model"""
        train_losses = []
        val_losses = []
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                total_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = self.criterion(val_outputs.squeeze(), y_val)
                
            avg_train_loss = total_loss / (len(X_train) // batch_size)
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss.item())
            
            self.scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        return train_losses, val_losses
    
    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()
    
    def predict_next_price(self, last_sequence, scaler, steps=1):
        """Predict next price(s)"""
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            pred = self.predict(current_sequence.reshape(1, *current_sequence.shape))
            predictions.append(pred[0][0])
            
            # Update sequence for next prediction
            new_row = current_sequence[-1].copy()
            new_row[0] = pred[0][0]  # Assuming price is first feature
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        return predictions
    
    def get_trading_signal(self, current_price, predicted_price, confidence_threshold=0.02):
        """Generate trading signal based on prediction"""
        price_change = (predicted_price - current_price) / current_price
        
        if price_change > confidence_threshold:
            return "BUY", abs(price_change) * 100, "Strong upward trend predicted"
        elif price_change < -confidence_threshold:
            return "SELL", abs(price_change) * 100, "Strong downward trend predicted"
        else:
            return "HOLD", abs(price_change) * 100, "Price expected to remain stable"
    
    def save_model(self, path):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])