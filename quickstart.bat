@echo off
title Simple RAG Quickstart
echo ===================================================
echo   Starting Simple RAG Backend and Frontend Servers
echo ===================================================

REM Start backend server
echo Launching Backend (FastAPI)...
start cmd /k "title Simple-RAG Backend && echo [Backend] Activating virtual environment... && .venv\Scripts\activate.bat && echo [Backend] Starting server... && python app.py"

REM Start frontend server
echo Launching Frontend (React)...
start cmd /k "title Simple-RAG Frontend && cd frontend && echo [Frontend] Starting web app... && npm start"

echo.
echo Both servers have been launched in separate terminal windows!
echo - Backend API: http://localhost:8000
echo - Frontend UI: http://localhost:3000
echo.
pause
