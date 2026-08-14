@echo off
chcp 65001 >nul
cd /d "%~dp0"
set FBREF_SSL_VERIFY=0
set TFF_QUIET=1

echo.
echo   TFF Fantezi Lig - Kadro Onerisi
echo   ------------------------------
echo   Lutfen bekleyiniz, analiz yapiliyor...
echo   (Bu islem 1-3 dakika surebilir)
echo.

python -m pip install -r requirements.txt -q 2>nul
python -m src.main --report-png data\weekly_report.png
echo.
if errorlevel 1 (
  echo Hata olustu. Detay icin: python -m src.main --verbose
  pause
) else (
  pause
)
