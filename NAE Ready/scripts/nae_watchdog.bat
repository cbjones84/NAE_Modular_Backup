@echo off
cd /d "C:\Users\v-nat\NAE\NAE Ready"
tasklist /FI "WINDOWTITLE eq NAE*" 2>NUL | find /I "python.exe" >NUL
if errorlevel 1 (
    echo [%date% %time%] NAE not running, starting... >> "C:\Users\v-nat\NAE\NAE Ready\logs\watchdog.log"
    start "NAE Autonomous Master" /MIN "C:\Users\v-nat\NAE\NAE Ready\scripts\run_nae_service.bat"
) else (
    echo [%date% %time%] NAE is running >> "C:\Users\v-nat\NAE\NAE Ready\logs\watchdog.log"
)
