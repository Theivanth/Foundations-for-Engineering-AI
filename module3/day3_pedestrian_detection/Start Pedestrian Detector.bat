@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%USERPROFILE%\.venvs\m1\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo Could not find the m1 Python environment:
    echo "%PYTHON%"
    echo.
    echo Open this project in VS Code and select the m1 interpreter,
    echo or update this launcher to point to your Python environment.
    pause
    exit /b 1
)

"%PYTHON%" "%~dp0predict.py"
if errorlevel 1 (
    echo.
    echo The detector stopped because of an error. See the message above.
    pause
)

endlocal
