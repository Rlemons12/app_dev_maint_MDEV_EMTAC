$ErrorActionPreference = "Stop"

$env:PATH = "E:\emtac\services\.venv_services\Lib\site-packages\pywin32_system32;E:\emtac\services\.venv_services\Lib\site-packages\win32;$env:PATH"

$env:PYTHONPATH = "E:\emtac\services\.venv_services\Lib\site-packages\win32\lib;E:\emtac\services\.venv_services\Lib\site-packages\win32;E:\emtac\services\.venv_services\Lib\site-packages;$env:PYTHONPATH"

Set-Location "E:\emtac\services\mcp_server"

& "E:\emtac\services\.venv_services\Scripts\python.exe" -m listed_server.mcp_coordinator.server