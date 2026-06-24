@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Updating from cloud...
git pull --no-edit 2>nul
echo Starting bot server in a new window...
start "Bot server - KEEP THIS OPEN" cmd /k python dashboard.py
timeout /t 6 >nul
start "" http://127.0.0.1:5000
echo.
echo Panel opened: http://127.0.0.1:5000
echo Keep the "Bot server" window open. Close it to stop the bot.
echo This window will close in 6 seconds.
timeout /t 6 >nul
