#!/usr/bin/env pwsh
<#
.SYNOPSIS
    NAE Windows Service Setup Script
    Configures NAE to run continuously as a Windows scheduled task with restart capabilities
.DESCRIPTION
    This script:
    1. Creates a scheduled task that runs at system startup
    2. Configures automatic restart on failure
    3. Sets power management to prevent sleep from stopping NAE
    4. Creates a watchdog task that ensures NAE is always running
#>

$ErrorActionPreference = "Stop"

# Configuration
$NAE_ROOT = "C:\Users\v-nat\NAE\NAE Ready"
$PYTHON_PATH = (Get-Command python).Source
$TASK_NAME = "NAE_Autonomous_Master"
$WATCHDOG_TASK_NAME = "NAE_Watchdog"
$LOG_DIR = "$NAE_ROOT\logs"

# Ensure log directory exists
if (-not (Test-Path $LOG_DIR)) {
    New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "NAE Windows Service Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Create the NAE launcher script
$LauncherScript = @"
@echo off
cd /d "$NAE_ROOT"
set PYTHONPATH=$NAE_ROOT
"$PYTHON_PATH" nae_autonomous_master.py >> "$LOG_DIR\nae_service.log" 2>&1
"@

$LauncherPath = "$NAE_ROOT\scripts\run_nae_service.bat"
Set-Content -Path $LauncherPath -Value $LauncherScript -Force
Write-Host "[OK] Created launcher script: $LauncherPath" -ForegroundColor Green

# Create watchdog script that checks if NAE is running and restarts if needed
$WatchdogScript = @"
@echo off
cd /d "$NAE_ROOT"
tasklist /FI "WINDOWTITLE eq NAE*" 2>NUL | find /I "python.exe" >NUL
if errorlevel 1 (
    echo [%date% %time%] NAE not running, starting... >> "$LOG_DIR\watchdog.log"
    start "NAE Autonomous Master" /MIN "$LauncherPath"
) else (
    echo [%date% %time%] NAE is running >> "$LOG_DIR\watchdog.log"
)
"@

$WatchdogPath = "$NAE_ROOT\scripts\nae_watchdog.bat"
Set-Content -Path $WatchdogPath -Value $WatchdogScript -Force
Write-Host "[OK] Created watchdog script: $WatchdogPath" -ForegroundColor Green

# Remove existing tasks if they exist
Write-Host ""
Write-Host "Configuring scheduled tasks..." -ForegroundColor Yellow

$existingTask = Get-ScheduledTask -TaskName $TASK_NAME -ErrorAction SilentlyContinue
if ($existingTask) {
    Unregister-ScheduledTask -TaskName $TASK_NAME -Confirm:$false
    Write-Host "[OK] Removed existing task: $TASK_NAME" -ForegroundColor Yellow
}

$existingWatchdog = Get-ScheduledTask -TaskName $WATCHDOG_TASK_NAME -ErrorAction SilentlyContinue
if ($existingWatchdog) {
    Unregister-ScheduledTask -TaskName $WATCHDOG_TASK_NAME -Confirm:$false
    Write-Host "[OK] Removed existing task: $WATCHDOG_TASK_NAME" -ForegroundColor Yellow
}

# Create the main NAE task that runs at startup
$Action = New-ScheduledTaskAction -Execute $LauncherPath -WorkingDirectory $NAE_ROOT
$Trigger = New-ScheduledTaskTrigger -AtStartup
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable:$false `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 365)

$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest

Register-ScheduledTask `
    -TaskName $TASK_NAME `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "NAE Autonomous Master Controller - Runs trading agents continuously" | Out-Null

Write-Host "[OK] Created scheduled task: $TASK_NAME" -ForegroundColor Green

# Create watchdog task that runs every 5 minutes
$WatchdogAction = New-ScheduledTaskAction -Execute $WatchdogPath -WorkingDirectory $NAE_ROOT
$WatchdogTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5)
$WatchdogSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 2)

Register-ScheduledTask `
    -TaskName $WATCHDOG_TASK_NAME `
    -Action $WatchdogAction `
    -Trigger $WatchdogTrigger `
    -Settings $WatchdogSettings `
    -Principal $Principal `
    -Description "NAE Watchdog - Ensures NAE is always running" | Out-Null

Write-Host "[OK] Created watchdog task: $WATCHDOG_TASK_NAME" -ForegroundColor Green

# Configure power settings to prevent sleep from stopping NAE
Write-Host ""
Write-Host "Configuring power settings..." -ForegroundColor Yellow

# Prevent system sleep when NAE is running (set to High Performance when plugged in)
try {
    # Disable sleep when plugged in
    powercfg /change standby-timeout-ac 0
    powercfg /change hibernate-timeout-ac 0
    Write-Host "[OK] Disabled sleep/hibernate when plugged in" -ForegroundColor Green
} catch {
    Write-Host "[WARN] Could not modify power settings (may need admin rights)" -ForegroundColor Yellow
}

# Allow wake timers
try {
    powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP RTCWAKE 1
    powercfg /setactive SCHEME_CURRENT
    Write-Host "[OK] Enabled wake timers" -ForegroundColor Green
} catch {
    Write-Host "[WARN] Could not enable wake timers" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "NAE Service Setup Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "NAE will:" -ForegroundColor White
Write-Host "  - Start automatically at system boot" -ForegroundColor White
Write-Host "  - Restart automatically if it crashes" -ForegroundColor White
Write-Host "  - Be monitored by watchdog every 5 minutes" -ForegroundColor White
Write-Host "  - Continue running during sleep (power settings adjusted)" -ForegroundColor White
Write-Host ""
Write-Host "To start NAE now, run:" -ForegroundColor Yellow
Write-Host "  Start-ScheduledTask -TaskName '$TASK_NAME'" -ForegroundColor Cyan
Write-Host ""

