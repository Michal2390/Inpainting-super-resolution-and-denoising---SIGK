#!/usr/bin/env powershell
# Skrypt do uruchamiania projektu SIGK w PowerShell

$PythonPath = "C:\Users\Michał\AppData\Local\Programs\Python\Python314\python.exe"
$ProjectPath = "C:\Users\Michał\PyCharmMiscProject"

# Sprawdzenie czy Python istnieje
if (-not (Test-Path $PythonPath)) {
    Write-Host "Błąd: Python nie znaleziony w $PythonPath" -ForegroundColor Red
    Write-Host "Zainstaluj Python lub zmień ścieżkę w tym pliku" -ForegroundColor Yellow
    Read-Host "Naciśnij Enter aby wyjść"
    exit 1
}

Write-Host "`n════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   PROJEKT SIGK - MODYFIKACJA OBRAZÓW" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════`n" -ForegroundColor Cyan

Write-Host "Znaleziony Python: $PythonPath" -ForegroundColor Green
Write-Host "`nWybierz akcję:`n"

Write-Host "  1) Zainstaluj zależności (requirements.txt)" -ForegroundColor Yellow
Write-Host "  2) Test weryfikacyjny (quickstart.py)" -ForegroundColor Yellow
Write-Host "  3) Trenowanie modeli (main.py)" -ForegroundColor Yellow
Write-Host "  4) Jupyter notebook (evaluation.ipynb)" -ForegroundColor Yellow
Write-Host "  5) Wyjście" -ForegroundColor Yellow
Write-Host "`n"

$Choice = Read-Host "Wpisz numer (1-5)"

switch ($Choice) {
    "1" {
        Write-Host "`nInstalowanie zależności..." -ForegroundColor Cyan
        & $PythonPath -m pip install -r "$ProjectPath\requirements.txt"
        Read-Host "`nNaciśnij Enter aby wyjść"
    }
    "2" {
        Write-Host "`nUruchamianie testów weryfikacyjnych..." -ForegroundColor Cyan
        Set-Location $ProjectPath
        & $PythonPath quickstart.py
        Read-Host "`nNaciśnij Enter aby wyjść"
    }
    "3" {
        Write-Host "`nUruchamianie trenowania modeli..." -ForegroundColor Cyan
        Write-Host "Uwaga: To może trwać kilka godzin!" -ForegroundColor Yellow
        Set-Location $ProjectPath
        & $PythonPath main.py
        Read-Host "`nNaciśnij Enter aby wyjść"
    }
    "4" {
        Write-Host "`nUruchamianie Jupyter notebook..." -ForegroundColor Cyan
        Set-Location $ProjectPath
        & $PythonPath -m jupyter notebook notebooks/evaluation.ipynb
    }
    "5" {
        exit 0
    }
    default {
        Write-Host "Nieznana opcja" -ForegroundColor Red
        Read-Host "Naciśnij Enter aby wyjść"
    }
}

