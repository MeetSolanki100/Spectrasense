@echo off
echo ================================
echo Starting Voice Assistant Backend
echo ================================

cd backend
call venv\Scripts\activate

echo.
echo Starting FastAPI server on http://localhost:8000
echo Press Ctrl+C to stop the server
echo.

python api.py

pause