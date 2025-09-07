@echo off
setlocal enableextensions enabledelayedexpansion

echo Upgrading pip tooling...
".\.venv\Scripts\python.exe" -m pip install -U pip setuptools wheel
if errorlevel 1 (
  echo Failed to upgrade pip tooling.
  exit /b 1
)

echo Installing requirements...
".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Failed to install requirements.
  exit /b 1
)

echo Dependencies installed successfully.
exit /b 0


