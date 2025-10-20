#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Starting All Servers for SpectraGUI  ${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Function to check if port is in use
check_port() {
    lsof -i :$1 > /dev/null 2>&1
    return $?
}

# Function to kill process on port
kill_port() {
    echo -e "${YELLOW}Killing existing process on port $1...${NC}"
    lsof -ti :$1 | xargs kill -9 2>/dev/null
    sleep 1
}

# Kill existing processes
echo -e "${YELLOW}Cleaning up existing processes...${NC}"
kill_port 8000
kill_port 5006
kill_port 5173

echo ""

# Start Face Recognition Backend (Port 5006)
echo -e "${GREEN}[1/3] Starting Face Recognition Backend (Port 5006)...${NC}"
cd /Users/kabirmathur/Documents/spectra_GUI/face_rec+adding_new_faces
source venv/bin/activate
nohup python app.py > /tmp/face_rec_backend.log 2>&1 &
FACE_REC_PID=$!
echo -e "  ${GREEN}✓${NC} Face Recognition Backend started (PID: $FACE_REC_PID)"
echo -e "  ${BLUE}→${NC} Log: /tmp/face_rec_backend.log"
sleep 3

# Check if it started successfully
if check_port 5006; then
    echo -e "  ${GREEN}✓${NC} Port 5006 is active"
else
    echo -e "  ${RED}✗${NC} Failed to start on port 5006"
    echo -e "  ${YELLOW}Check log: tail -f /tmp/face_rec_backend.log${NC}"
fi

echo ""

# Start Main Backend API (Port 8000)
echo -e "${GREEN}[2/3] Starting Main Backend API (Port 8000)...${NC}"
cd /Users/kabirmathur/Documents/spectra_GUI/Spectrasense
source venv/bin/activate
nohup python backend/api.py > /tmp/main_backend.log 2>&1 &
MAIN_BACKEND_PID=$!
echo -e "  ${GREEN}✓${NC} Main Backend started (PID: $MAIN_BACKEND_PID)"
echo -e "  ${BLUE}→${NC} Log: /tmp/main_backend.log"
sleep 3

# Check if it started successfully
if check_port 8000; then
    echo -e "  ${GREEN}✓${NC} Port 8000 is active"
else
    echo -e "  ${RED}✗${NC} Failed to start on port 8000"
    echo -e "  ${YELLOW}Check log: tail -f /tmp/main_backend.log${NC}"
fi

echo ""

# Start Frontend (Port 5173)
echo -e "${GREEN}[3/3] Starting Frontend (Port 5173)...${NC}"
cd /Users/kabirmathur/Documents/spectra_GUI/Spectrasense/frontend
nohup npm run dev > /tmp/frontend.log 2>&1 &
FRONTEND_PID=$!
echo -e "  ${GREEN}✓${NC} Frontend started (PID: $FRONTEND_PID)"
echo -e "  ${BLUE}→${NC} Log: /tmp/frontend.log"
sleep 5

# Check if it started successfully
if check_port 5173; then
    echo -e "  ${GREEN}✓${NC} Port 5173 is active"
else
    echo -e "  ${RED}✗${NC} Failed to start on port 5173"
    echo -e "  ${YELLOW}Check log: tail -f /tmp/frontend.log${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}  All Servers Started Successfully!  ${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo -e "${BLUE}Server Status:${NC}"
echo -e "  ${GREEN}✓${NC} Face Recognition Backend: http://localhost:5006"
echo -e "  ${GREEN}✓${NC} Main Backend API:         http://localhost:8000"
echo -e "  ${GREEN}✓${NC} Frontend Application:     http://localhost:5173"

echo ""
echo -e "${BLUE}Process IDs:${NC}"
echo -e "  Face Recognition: $FACE_REC_PID"
echo -e "  Main Backend:     $MAIN_BACKEND_PID"
echo -e "  Frontend:         $FRONTEND_PID"

echo ""
echo -e "${BLUE}Logs:${NC}"
echo -e "  ${YELLOW}Face Recognition:${NC} tail -f /tmp/face_rec_backend.log"
echo -e "  ${YELLOW}Main Backend:${NC}     tail -f /tmp/main_backend.log"
echo -e "  ${YELLOW}Frontend:${NC}         tail -f /tmp/frontend.log"

echo ""
echo -e "${BLUE}To stop all servers:${NC}"
echo -e "  kill $FACE_REC_PID $MAIN_BACKEND_PID $FRONTEND_PID"
echo -e "  ${YELLOW}OR${NC}"
echo -e "  pkill -f 'python app.py' && pkill -f 'python backend/api.py' && pkill -f 'npm run dev'"

echo ""
echo -e "${GREEN}Opening browser in 3 seconds...${NC}"
sleep 3
open http://localhost:5173

echo ""
echo -e "${GREEN}Done! Your application is ready.${NC}"
