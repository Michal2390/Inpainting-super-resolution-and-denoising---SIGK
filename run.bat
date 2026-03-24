@echo off
REM Skrypt do uruchamiania projektu SIGK
REM Ścieżka do Pythona - spróbuj Python 3.12 (GPU), jeśli nie ma - użyj Python 3.14 (CPU)
python3 --version >nul 2>&1
if %ERRORLEVEL% == 0 (
    set PYTHON_PATH=python3
    echo [OK] Uzywam Python 3.12 (GPU)
) else (
    set PYTHON_PATH=C:\Users\Michał\AppData\Local\Programs\Python\Python314\python.exe
    echo [OK] Python 3.12 nie znaleziony, uzywam Python 3.14 (CPU)
)
set PROJECT_PATH=C:\Users\Michał\PyCharmMiscProject

REM Sprawdzenie czy Python istnieje
if not exist "%PYTHON_PATH%" (
    echo Błąd: Python nie znaleziony w %PYTHON_PATH%
    echo Zainstaluj Python lub zmień ścieżkę w tym pliku
    pause
    exit /b 1
)

echo.
echo ════════════════════════════════════════════════════════════════
echo   PROJEKT SIGK - MODYFIKACJA OBRAZÓW
echo ════════════════════════════════════════════════════════════════
echo.
echo Znaleziony Python: %PYTHON_PATH%
echo.
echo Wybierz akcję:
echo 1) Zainstaluj zależności (requirements.txt)
echo 2) Test weryfikacyjny (quickstart.py)
echo 2b) Test GPU (test_gpu.py) - RTX 4070Ti
echo 3) Trenowanie modeli (main.py)
echo 4) Jupyter notebook (evaluation.ipynb)
echo 5) Wyjście
echo.

set /p CHOICE="Wpisz numer (1-5): "

if "%CHOICE%"=="1" (
    echo.
    echo Instalowanie zależności...
    "%PYTHON_PATH%" -m pip install -r "%PROJECT_PATH%\requirements.txt"
    pause
)

if "%CHOICE%"=="2" (
    echo.
    echo Uruchamianie testów weryfikacyjnych...
    cd /d "%PROJECT_PATH%"
    "%PYTHON_PATH%" quickstart.py
    pause
)

if "%CHOICE%"=="2b" (
    echo.
    echo Uruchamianie testu GPU...
    cd /d "%PROJECT_PATH%"
    "%PYTHON_PATH%" test_gpu.py
    pause
)

if "%CHOICE%"=="3" (
    echo.
    echo Uruchamianie trenowania modeli...
    echo Uwaga: To może trwać kilka godzin!
    cd /d "%PROJECT_PATH%"
    "%PYTHON_PATH%" main.py
    pause
)

if "%CHOICE%"=="4" (
    echo.
    echo Uruchamianie Jupyter notebook...
    cd /d "%PROJECT_PATH%"
    "%PYTHON_PATH%" -m jupyter notebook notebooks/evaluation.ipynb
)

if "%CHOICE%"=="5" (
    exit /b 0
)

echo Nieznana opcja
pause
goto start





