#!/usr/bin/env python3
"""
Quick Start Script for Bitcoin Price Predictor
This script provides a simple way to get started quickly.
"""

import os
import sys
import subprocess
import time

def run_command(command, cwd=None):
    """Run a command and return success status"""
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"SUCCESS: {command}")
            return True
        else:
            print(f"FAILED: {command}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"EXCEPTION running {command}: {e}")
        return False

def main():
    print("Bitcoin AI Predictor - Quick Start")
    print("=" * 50)
    
    # Check if dataset exists
    if not os.path.exists("dataset.csv"):
        print("ERROR: dataset.csv not found!")
        print("Please ensure your Bitcoin dataset is in the root directory.")
        return
    
    print("Dataset found!")
    
    # Install backend dependencies
    print("\n1. Installing Python dependencies...")
    if not run_command(f"{sys.executable} -m pip install pandas numpy scikit-learn torch matplotlib seaborn flask flask-cors yfinance joblib ta"):
        print("Failed to install Python dependencies")
        return
    
    # Train simple model
    print("\n2. Training AI model on your data...")
    if not run_command(f"{sys.executable} backend/train_simple.py"):
        print("Model training failed. Check your dataset format.")
        return
    
    # Install frontend dependencies
    print("\n3. Setting up frontend...")
    if not run_command("npm install", cwd="frontend"):
        print("Frontend setup failed. Make sure Node.js is installed.")
        return
    
    print("\nSetup completed successfully!")
    print("\nStarting the application...")
    
    # Start backend in background
    print("Starting backend server...")
    backend_process = subprocess.Popen([sys.executable, "api/simple_app.py"], cwd="backend")
    
    # Wait a moment for backend to start
    time.sleep(3)
    
    # Start frontend
    print("Starting frontend server...")
    print("Opening http://localhost:3000 in your browser...")
    
    try:
        subprocess.run(["npm", "run", "dev"], cwd="frontend")
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend_process.terminate()
        print("Servers stopped.")

if __name__ == "__main__":
    main()