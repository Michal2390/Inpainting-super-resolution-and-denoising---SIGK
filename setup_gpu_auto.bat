@echo off
REM Automatyczna instalacja Python 3.12 + PyTorch CUDA dla RTX 4070Ti

setlocal enabledelayedexpansion

echo.
echo ====================================================================
echo  INSTALACJA PYTHON 3.12 + PyTorch CUDA DLA RTX 4070Ti
echo ====================================================================
echo.

REM Ścieżka do Python 3.12 installer
set PYTHON312_EXE=C:\Users\Michał\Downloads\python-3.12.2-amd64.exe

REM Sprawdź czy installer istnieje
if not exist "%PYTHON312_EXE%" (
    echo [ERROR] Nie znaleziono: %PYTHON312_EXE%
    echo.
    echo Pobierz Python 3.12 ze strony:
    echo https://www.python.org/downloads/release/python-3122/
    echo.
    pause
    exit /b 1
)

echo [KROK 1] Instalowanie Python 3.12...
echo.
echo Postępuj zgodnie z instrukcjami instalatora:
echo   - Zaznacz: [x] Add Python 3.12 to PATH
echo   - Zaznacz: [x] Install for all users (jeśli pytany)
echo   - Kliknij: Install Now
echo.
echo Czekaj na zakończenie instalacji...
echo.
echo Naciśnij dowolny klawisz aby kontynuować...
pause

REM Uruchom installer
"%PYTHON312_EXE%" /quiet InstallAllUsers=1 PrependPath=1

REM Czekaj na koniec
timeout /t 10 /nobreak

REM Sprawdź czy Python 3.12 został zainstalowany
echo.
echo [KROK 2] Weryfikacja instalacji Python 3.12...

C:\Python312\python.exe --version >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo [OK] Python 3.12 zainstalowany!
    C:\Python312\python.exe --version
) else (
    echo [ERROR] Problem z instalacją Python 3.12
    echo.
    echo Spróbuj zainstalować ręcznie:
    echo 1. Otwórz: %PYTHON312_EXE%
    echo 2. Zaznacz: Add Python 3.12 to PATH
    echo 3. Kliknij: Install Now
    echo.
    pause
    exit /b 1
)

REM Zainstaluj PyTorch CUDA
echo.
echo [KROK 3] Instalowanie PyTorch z CUDA 12.1 support...
echo.
echo To może potrwać kilka minut...
echo.

C:\Python312\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade

if %ERRORLEVEL% == 0 (
    echo.
    echo [OK] PyTorch zainstalowany!
) else (
    echo.
    echo [ERROR] Problem z instalacją PyTorch
    echo.
    echo Spróbuj ręcznie:
    echo C:\Python312\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo.
    pause
    exit /b 1
)

REM Zmień run.bat
echo.
echo [KROK 4] Konfiguracja run.bat...
echo.

set PROJECT_DIR=C:\Users\Michał\PyCharmMiscProject
set RUN_BAT=%PROJECT_DIR%\run.bat

REM Utwórz backup
copy "%RUN_BAT%" "%RUN_BAT%.backup" >nul

REM Zmień PYTHON_PATH
powershell -Command "(Get-Content '%RUN_BAT%') -replace 'Python314', 'Python312' | Set-Content '%RUN_BAT%'"

echo [OK] run.bat zaktualizowany!

REM Test GPU
echo.
echo [KROK 5] Test GPU...
echo.

cd /d "%PROJECT_DIR%"
C:\Python312\python.exe test_gpu.py

echo.
echo ====================================================================
echo  GOTOWE!
echo ====================================================================
echo.
echo Teraz możesz uruchomić trenowanie:
echo   run.bat → "3" → Trenowanie
echo.
echo Trenowanie będzie trwać ~6 godzin zamiast 30!
echo.
pause

