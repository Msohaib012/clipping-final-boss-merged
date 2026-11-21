@echo off
title CFB Clipper
python --version >nul 2>&1
if errorlevel 1 (
  echo Python nahi mila. Python 3.12 install karo: https://www.python.org/downloads/
  echo Install karte waqt "Add Python to PATH" zaroor tick karna.
  pause
  exit /b 1
)
ffmpeg -version >nul 2>&1
if errorlevel 1 (
  echo FFmpeg nahi mila. PowerShell kholo aur ye chalao: winget install Gyan.FFmpeg
  echo Phir ye window band karke start.bat dobara chalao.
  pause
  exit /b 1
)
python run.py
pause
