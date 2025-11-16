#!/bin/bash

# TicTacToe RL Agent Startup Script
# This script helps you get the application running quickly

echo "🎮 TicTacToe Reinforcement Learning Agent"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ first."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

echo "✅ Prerequisites check passed!"
echo ""

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install Python dependencies"
    exit 1
fi

echo "✅ Python dependencies installed!"
echo ""

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
cd frontend
npm install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install Node.js dependencies"
    exit 1
fi

echo "✅ Node.js dependencies installed!"
echo ""

# Go back to root directory
cd ..

echo "🚀 Starting the application..."
echo ""
echo "The application will start in two parts:"
echo "1. Backend server (Flask) - http://localhost:5000"
echo "2. Frontend server (React) - http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start backend server in background
echo "🔧 Starting backend server..."
python3 app.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend server in background
echo "🎨 Starting frontend server..."
cd frontend
npm start &
FRONTEND_PID=$!

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
