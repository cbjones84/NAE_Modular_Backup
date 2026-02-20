@echo off
cd /d "C:\Users\v-nat\NAE\NAE Ready"
set PYTHONPATH=C:\Users\v-nat\NAE\NAE Ready
"C:\Users\v-nat\AppData\Local\Programs\Python\Python311\python.exe" nae_autonomous_master.py >> "C:\Users\v-nat\NAE\NAE Ready\logs\nae_service.log" 2>&1
