@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Подтягиваю свежее обучение из облака...
git pull --no-edit 2>nul
echo Запускаю панель управления ботом...
start "Торговый бот" /min python dashboard.py
timeout /t 3 /nobreak >nul
start "" http://127.0.0.1:5000
echo.
echo Панель открыта в браузере: http://127.0.0.1:5000
echo Это окно можно свернуть. Чтобы выключить бота - закрой это окно.
echo.
pause
