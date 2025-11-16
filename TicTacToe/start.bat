@echo off
REM TicTacToe RL Agent Startup Script for Windows
REM This script helps you get the application running quickly

echo 🎮 TicTacToe Reinforcement Learning Agent
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is not installed. Please install Node.js 16+ first.
    pause
    exit /b 1
)

REM Check if npm is installed
npm --version >nul 2>&1
if errorlevel 1 (
    echo ❌ npm is not installed. Please install npm first.
    pause
    exit /b 1
)

echo ✅ Prerequisites check passed!
echo.

REM Install Python dependencies
echo 📦 Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install Python dependencies
    pause
    exit /b 1
)

echo ✅ Python dependencies installed!
echo.

REM Install Node.js dependencies
echo 📦 Installing Node.js dependencies...
cd frontend
npm install
if errorlevel 1 (
    echo ❌ Failed to install Node.js dependencies
    pause
    exit /b 1
)

echo ✅ Node.js dependencies installed!
echo.

REM Go back to root directory
cd ..

echo 🚀 Starting the application...
echo.
echo The application will start in two parts:
echo 1. Backend server (Flask) - http://localhost:5001
echo 2. Frontend server (React) - http://localhost:3000
echo.
echo Press Ctrl+C to stop both servers
echo.

REM Start backend server
echo 🔧 Starting backend server...
start "Backend Server" cmd /k "python app.py"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend server
echo 🎨 Starting frontend server...
cd frontend
start "Frontend Server" cmd /k "npm start"

echo.
echo ✅ Both servers are starting up!
echo 🌐 Open http://localhost:3000 in your browser to play!
echo.
pause
