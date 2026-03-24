@echo off
REM Automatic PyTorch CUDA installation after Python 3.12 is installed

setlocal enabledelayedexpansion

echo.
echo ====================================================================
echo  CZEKAM NA ZAINSTALOWANIE PYTHON 3.12
echo ====================================================================
echo.
echo Powinno się otworzyć okno instalacyjne.
echo.
echo WAŻNE: Zaznacz "Add Python 3.12 to PATH" przed kliknięciem Install!
echo.
echo Czekam...
echo.

REM Czekaj aż Python 3.12 zostanie zainstalowany
:WAIT_FOR_PYTHON312
timeout /t 5 >nul
python3 --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Czekam na Python 3.12...
    goto WAIT_FOR_PYTHON312
)

REM Python 3.12 znaleziony!
echo.
echo [OK] Python 3.12 zainstalowany!
echo.
python3 --version
echo.

REM Zainstaluj PyTorch CUDA
echo.
echo ====================================================================
echo  INSTALACJA PyTorch Z CUDA 12.1 SUPPORT
echo ====================================================================
echo.
echo Instaluję PyTorch CUDA dla RTX 4070Ti...
echo To może potrwać kilka minut...
echo.

python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade

if %ERRORLEVEL% == 0 (
    echo.
    echo [OK] PyTorch zainstalowany!
    echo.
) else (
    echo.
    echo [ERROR] Problem z instalacją PyTorch!
    echo.
    pause
    exit /b 1
)

REM Test GPU
echo.
echo ====================================================================
echo  TEST GPU - RTX 4070Ti
echo ====================================================================
echo.

cd /d "C:\Users\Michał\PyCharmMiscProject"
python3 test_gpu.py

REM Zmień run.bat
echo.
echo ====================================================================
echo  AKTUALIZACJA run.bat
echo ====================================================================
echo.

powershell -Command "(Get-Content 'C:\Users\Michał\PyCharmMiscProject\run.bat') -replace 'Python314', 'python3' | Set-Content 'C:\Users\Michał\PyCharmMiscProject\run.bat'"

echo [OK] run.bat zaktualizowany!

echo.
echo ====================================================================
echo  GOTOWE! PROJEKT JEST NA GPU!
echo ====================================================================
echo.
echo Teraz uruchom:
echo   run.bat
echo.
echo Wybierz:
echo   3) Trenowanie modeli
echo.
echo Trenowanie będzie trwać ~6 godzin zamiast 30!
echo.

pause

