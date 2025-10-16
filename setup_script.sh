#!/bin/bash

echo "================================"
echo "Voice Assistant Setup Script"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is installed
echo -e "\n${YELLOW}Checking Python installation...${NC}"
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python is not installed. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

PYTHON_CMD=$(command -v python3 || command -v python)
echo -e "${GREEN}Python found: $PYTHON_CMD${NC}"

# Check if Node.js is installed
echo -e "\n${YELLOW}Checking Node.js installation...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${RED}Node.js is not installed. Please install Node.js 16 or higher.${NC}"
    exit 1
fi
echo -e "${GREEN}Node.js found: $(node --version)${NC}"

# Create directory structure
echo -e "\n${YELLOW}Creating directory structure...${NC}"
mkdir -p backend
mkdir -p frontend
mkdir -p components
mkdir -p models/whisper
mkdir -p chroma_db

# Backend setup
echo -e "\n${YELLOW}Setting up backend...${NC}"
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install Python dependencies
echo -e "\n${YELLOW}Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r ../requirements.txt

# Check if Ollama is running
echo -e "\n${YELLOW}Checking Ollama...${NC}"
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo -e "${RED}Ollama is not running. Please start Ollama and ensure llama3.1:8b model is installed.${NC}"
    echo -e "${YELLOW}Install with: ollama pull llama3.1:8b${NC}"
else
    echo -e "${GREEN}Ollama is running${NC}"
fi

cd ..

# Frontend setup
echo -e "\n${YELLOW}Setting up frontend...${NC}"
cd frontend

# Initialize React app if package.json doesn't exist
if [ ! -f "package.json" ]; then
    echo "Initializing React application..."
    npm create vite@latest . -- --template react
fi

# Install dependencies
echo "Installing Node.js dependencies..."
npm install
npm install lucide-react

cd ..

echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${GREEN}================================${NC}"
echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Ensure Ollama is running with llama3.1:8b model"
echo "2. Start the backend: cd backend && source venv/bin/activate && python api.py"
echo "3. Start the frontend: cd frontend && npm run dev"
echo "4. Open http://localhost:5173 in your browser"
echo ""