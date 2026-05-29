@echo off
cd /d "%~dp0"
echo Starting hh-bot...
python main.py >> bot.log 2>&1
echo.
echo Bot stopped. Check bot.log for details.
pause
