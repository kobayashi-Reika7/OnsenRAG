@echo off
REM Day4 フロントエンド起動スクリプト（npm がパスにない場合用）
set "PATH=C:\Program Files\nodejs;%PATH%"
cd /d "%~dp0"
call npm run dev
pause
