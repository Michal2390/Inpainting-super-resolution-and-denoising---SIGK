@echo off
REM Skrypt do pobrania i instalacji Python 3.12

echo.
echo ====================================================================
echo  Pobranie Python 3.12 do PyTorch CUDA support
echo ====================================================================
echo.

REM Pobierz Python 3.12 installer
echo Pobieranie Python 3.12.2...
powershell -Command "(New-Object System.Net.ServicePointManager).SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.2/python-3.12.2-amd64.exe' -OutFile '%TEMP%\python-3.12.2-amd64.exe'"

echo.
echo Instalowanie Python 3.12...
echo Postępuj zgodnie z instrukcjami instalatora:
echo   - Zaznacz: Add Python 3.12 to PATH
echo   - Zaznacz: Install for all users (opcjonalnie)
echo   - Kliknij: Install Now
echo.

"%TEMP%\python-3.12.2-amd64.exe"

echo.
echo ====================================================================
echo  Python 3.12 został zainstalowany!
echo ====================================================================
echo.
echo Teraz zainstaluj PyTorch CUDA:
echo.
echo C:\Python312\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo.
echo Następnie uruchom: run.bat
echo (i zmień PYTHON_PATH na: C:\Python312\python.exe)
echo.

pause

