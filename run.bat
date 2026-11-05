@echo off
title Face Engine - Streamlit Web Dashboard
cd /d "%~dp0"
echo ====================================================
echo Starting Face Engine Web Application
echo Local URL: http://localhost:8501
echo Compute Hardware: NVIDIA GeForce GTX 1660 SUPER
echo ====================================================
streamlit run app.py
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    pause
)
