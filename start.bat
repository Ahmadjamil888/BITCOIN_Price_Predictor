@echo off
echo Bitcoin AI Predictor - Quick Start
echo ===================================

REM Check if dataset exists
if not exist "dataset.csv" (
    echo ERROR: dataset.csv not found!
    echo Please ensure your Bitcoin dataset is in the root directory.
    pause
    exit /b 1
)

echo Dataset found!

REM Install Python dependencies
echo.
echo Installing Python dependencies...
pip install -r requirements.txt

REM Train model
echo.
echo Training AI model on your data...
cd backend
python train_simple.py
if errorlevel 1 (
    echo Model training failed!
    pause
    exit /b 1
)
cd ..

REM Install frontend dependencies
echo.
echo Setting up frontend...
cd frontend
call npm install
if errorlevel 1 (
    echo Frontend setup failed!
    pause
    exit /b 1
)

echo.
echo Setup completed successfully!
echo.
echo Starting backend server...
cd ..\backend
start "Backend Server" python api/simple_app.py

echo Waiting for backend to start...
timeout /t 3 /nobreak > nul

echo Starting frontend server...
cd ..\frontend
echo Opening http://localhost:3000 in your browser...
npm run dev

pause