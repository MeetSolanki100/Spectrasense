#!/bin/bash

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Stopping all servers...${NC}\n"

# Kill processes on specific ports
echo -e "${RED}Killing Face Recognition Backend (Port 5006)...${NC}"
lsof -ti :5006 | xargs kill -9 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "  ${GREEN}✓${NC} Stopped"
else
    echo -e "  ${YELLOW}→${NC} No process found"
fi

echo -e "${RED}Killing Main Backend API (Port 8000)...${NC}"
lsof -ti :8000 | xargs kill -9 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "  ${GREEN}✓${NC} Stopped"
else
    echo -e "  ${YELLOW}→${NC} No process found"
fi

echo -e "${RED}Killing Frontend (Port 5173)...${NC}"
lsof -ti :5173 | xargs kill -9 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "  ${GREEN}✓${NC} Stopped"
else
    echo -e "  ${YELLOW}→${NC} No process found"
fi

# Also kill by process name as backup
pkill -f "python app.py" 2>/dev/null
pkill -f "python backend/api.py" 2>/dev/null
pkill -f "npm run dev" 2>/dev/null

echo ""
echo -e "${GREEN}All servers stopped.${NC}"
