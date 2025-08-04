#!/usr/bin/env python3
"""
Bitcoin Price Predictor Setup Script
This script sets up the environment and trains the initial model.
"""

import os
import sys
import subprocess
import pandas as pd

def install_requirements():
    """Install Python requirements"""
    print("Installing Python requirements...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt"])

def setup_directories():
    """Create necessary directories"""
    directories = [
        "backend/saved_models",
        "backend/data",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def check_dataset():
    """Check if dataset exists and is readable"""
    if not os.path.exists("dataset.csv"):
        print("ERROR: dataset.csv not found!")
        print("Please ensure your Bitcoin dataset is in the root directory as 'dataset.csv'")
        return False
    
    try:
        df = pd.read_csv("dataset.csv", nrows=5)
        print(f"Dataset found with columns: {list(df.columns)}")
        return True
    except Exception as e:
        print(f"ERROR reading dataset: {e}")
        return False

def train_initial_model():
    """Train the initial model"""
    print("Training initial model...")
    print("This may take a few minutes...")
    
    try:
        os.chdir("backend")
        
        # Try simple training first
        print("Using simple LSTM approach for better compatibility...")
        result = subprocess.run([sys.executable, "train_simple.py"], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Model training completed successfully!")
            print("Model saved to backend/saved_models/")
            print(result.stdout)
        else:
            print("Simple model training failed, trying advanced model...")
            print(result.stderr)
            
            # Fallback to advanced model
            result2 = subprocess.run([sys.executable, "train_model.py"], capture_output=True, text=True)
            if result2.returncode == 0:
                print("Advanced model training completed!")
                return True
            else:
                print("Both training approaches failed:")
                print(result2.stderr)
                return False
            
    except Exception as e:
        print(f"Error during training: {e}")
        return False
    finally:
        os.chdir("..")
    
    return True

def setup_frontend():
    """Setup frontend dependencies"""
    print("Setting up frontend...")
    
    try:
        os.chdir("frontend")
        
        # Install npm dependencies
        print("Installing npm dependencies...")
        result = subprocess.run(["npm", "install"], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("npm install failed:")
            print(result.stderr)
            return False
            
        print("Frontend setup completed!")
        
    except Exception as e:
        print(f"Error setting up frontend: {e}")
        return False
    finally:
        os.chdir("..")
    
    return True

def main():
    """Main setup function"""
    print("🚀 Bitcoin Price Predictor Setup")
    print("=" * 40)
    
    # Check dataset
    if not check_dataset():
        return
    
    # Setup directories
    setup_directories()
    
    # Install Python requirements
    install_requirements()
    
    # Train initial model
    if not train_initial_model():
        print("Setup incomplete due to training failure")
        return
    
    # Setup frontend
    if not setup_frontend():
        print("Setup incomplete due to frontend setup failure")
        return
    
    print("\n✅ Setup completed successfully!")
    print("\nTo start the application:")
    print("1. Backend: cd backend && python api/simple_app.py")
    print("2. Frontend: cd frontend && npm run dev")
    print("\nThen open http://localhost:3000 in your browser")
    print("\nNote: Using simple_app.py for better compatibility with your dataset")

if __name__ == "__main__":
    main()