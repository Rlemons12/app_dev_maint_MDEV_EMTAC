@echo off
echo Starting EMTAC Service Dashboard...
set EMTAC_ENV_PATH=E:\emtac\dev_env\.env
cd /d "E:\emtac\services\service_dashboard"
"E:\emtac\services\gpu\.venv_gpu\Scripts\python.exe" "app_services_dashboard.py"
pause