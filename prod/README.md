# EMTAC Production Blue/Green Deployment Notes

## Purpose

This folder contains the production blue/green deployment setup for EMTAC.

The production root is:

```text
E:\emtac\prod
```

The goal is to keep production slots clean, repeatable, and easy to switch between.

A production slot should contain only the runtime app files needed to serve EMTAC. It should not contain development files, test folders, duplicate databases, local logs, or generated cleanup artifacts.

---

## Recommended Production Layout

```text
E:\emtac\prod
├── README.md
├── slot_exclusions.txt
├── run
│   ├── green.pid
│   └── blue.pid
├── scripts
│   ├── start_emtac_slot.ps1
│   ├── stop_emtac_slot.ps1
│   ├── test_emtac_slot.ps1
│   ├── set_active_port.ps1
│   ├── freeze_emtac_slot.ps1
│   ├── audit_emtac_dependencies.ps1
│   └── copy_pgvector_from_dev.ps1
├── shared
│   ├── logs
│   ├── uploads
│   ├── temp
│   ├── backups
│   ├── cleanup_quarantine
│   ├── dependency_records
│   └── secure_env_backups
└── slots
    ├── blue
    └── green
```

---

## Port Plan

Recommended port layout:

```text
blue slot     -> 8101
green slot    -> 8102
tablet/active -> 5000
```

The tablet currently expects EMTAC on port `5000`.

Do not run the app directly on port `5000` if you want clean blue/green switching. Instead, run each slot on its own port, then forward port `5000` to the active slot.

Example:

```text
tablet -> http://172.19.194.129:5000/
5000   -> forwarded to green 8102
green  -> http://172.19.194.129:8102/
```

---

## Important Concept

The Flask app does not need to know every port.

The server layer controls ports:

```text
Waitress / portproxy -> listens on host:port
Flask app            -> handles requests
```

For WSGI startup, this deployment uses:

```text
wsgi.py -> ai_emtac.create_app()
```

The current `ai_emtac.py` uses a `create_app()` factory, so importing `ai_emtac` will not expose `ai_emtac.app` directly. That is expected.

---

## Shared Production Stores

The slot should not own mutable production data.

Use these shared/root locations:

```text
Production database/document store:
E:\emtac\Database

Production uploads:
E:\emtac\prod\shared\uploads

Production logs:
E:\emtac\prod\shared\logs

Production cleanup quarantine:
E:\emtac\prod\shared\cleanup_quarantine

Production dependency records:
E:\emtac\prod\shared\dependency_records

Production environment backups:
E:\emtac\prod\shared\secure_env_backups
```

The slot path below should be a junction only:

```text
E:\emtac\prod\slots\green\Database -> E:\emtac\Database
E:\emtac\prod\slots\blue\Database  -> E:\emtac\Database
```

Do not keep a physical duplicate `Database` folder inside a slot.

---

## Expected Clean Slot Root

A clean production slot should look roughly like this:

```text
E:\emtac\prod\slots\green
├── .venv
├── blueprints
├── Database          # junction to E:\emtac\Database
├── modules
├── static
├── templates
├── utilities
├── .env
├── ai_emtac.py
├── requirements.green.lock.txt
├── wsgi.py
└── __init__.py
```

The slot should not contain development/test folders such as:

```text
.idea
.vs
data
databases
load_process
logs
log_backup
mi_au_maint
pgvector
plugins
project_documentation
project_tests
scripts
scripts_output
sql
tests
test_config
utility_tools
_internal
__pycache__
```

The slot should also not contain loose development files such as:

```text
dvc.exe
unins000.exe
unins000.dat
mi_au_maint.zip
app.log
check_excel_headers.py
create_help_chat_files.py
postgres_control.py
verify_ingestion_pathway.py
deployment_manifest*.json
pip_check_*.txt
```

Use quarantine before deleting anything permanently.

---

## Slot Exclusion List

Deployment exclusions are documented here:

```text
E:\emtac\prod\slot_exclusions.txt
```

These exclusions should be applied when copying code from development into a blue or green slot.

Important exclusions include:

```text
.env
.env.*
.venv
Database(can be removed)
data
databases
load_process
logs
log_backup
mi_au_maint
pgvector
plugins
project_documentation
project_tests
scripts
scripts_output
sql
tests
test_config
utility_tools
_internal
__pycache__
requirements*.txt
deployment_manifest*.json
pip_check_*.txt
```

Do not exclude:

```text
ai_emtac.py
wsgi.py
blueprints
modules
static
templates
utilities
```

---

## Production `.env` Notes

The production slot `.env` lives here:

```text
E:\emtac\prod\slots\green\.env
E:\emtac\prod\slots\blue\.env
```

Do not commit `.env`.

Production values should include:

```env
APP_ENV=production
FLASK_ENV=production
FLASK_DEBUG=0
EMTAC_ENV=production
EMTAC_SLOT=green

HOST=0.0.0.0
PORT=8102

PROJECT_PATH=E:\emtac\prod\slots\green
VENV_PATH=E:\emtac\prod\slots\green\.venv
PYTHON_EXE=E:\emtac\prod\slots\green\.venv\Scripts\python.exe

DATABASE_DIR=E:\emtac\Database
EMTAC_UPLOAD_FOLDER=E:\emtac\prod\shared\uploads
EMTAC_LOG_DIR=E:\emtac\prod\shared\logs
LOGS_DIR=E:\emtac\prod\shared\logs
LOG_FILE=E:\emtac\prod\shared\logs\green_app.log
```

Offline production should not prompt for API keys. If the app is running offline and does not need external API calls, use non-empty placeholders:

```env
OPENAI_API_KEY=offline-disabled
ANTHROPIC_API_KEY=offline-disabled
HUGGINGFACE_API_KEY=offline-disabled
auth_token=offline-disabled
```

Leaving these blank can cause startup to block with:

```text
Enter your OpenAI API key:
```

That will prevent Waitress from becoming available on the slot port.

---

## Start Green on the LAN

Run from PowerShell:

```powershell
cd E:\emtac\prod

.\scripts\start_emtac_slot.ps1 -Slot green -Port 8102 -HostAddress 0.0.0.0 -Force
```

If running as Administrator and you want the script to add the firewall rule:

```powershell
.\scripts\start_emtac_slot.ps1 -Slot green -Port 8102 -HostAddress 0.0.0.0 -OpenFirewall -Force
```

The start script should:

1. Verify the slot folder.
2. Verify the slot venv Python.
3. Verify Waitress is installed.
4. Start Waitress on the requested port.
5. Write the PID file.
6. Wait for `/health` to return `200`.
7. Write logs to `E:\emtac\prod\shared\logs`.

---

## Stop Green

```powershell
cd E:\emtac\prod

.\scripts\stop_emtac_slot.ps1 -Slot green
```

The stop script should remove stale PID files and stop any process listening on the slot port.

For green, the slot port is:

```text
8102
```

For blue, the slot port is:

```text
8101
```

---

## Test Green

```powershell
cd E:\emtac\prod

.\scripts\test_emtac_slot.ps1 -HostName 127.0.0.1 -Port 8102
.\scripts\test_emtac_slot.ps1 -HostName 172.19.194.129 -Port 8102
```

Expected:

```text
/health     -> 200 OK
/api-status -> 200 OK
```

---

## Point Tablet Port 5000 to Green

Run PowerShell as Administrator:

```powershell
cd E:\emtac\prod

.\scripts\set_active_port.ps1 -ActivePort 5000 -TargetPort 8102 -OpenFirewall
```

Tablet URL:

```text
http://172.19.194.129:5000/
```

Check portproxy:

```powershell
netsh interface portproxy show v4tov4
```

Expected mapping:

```text
0.0.0.0  5000  ->  127.0.0.1  8102
```

---

## Switch Tablet Port 5000 to Blue Later

If blue is running on `8101`:

```powershell
cd E:\emtac\prod

.\scripts\set_active_port.ps1 -ActivePort 5000 -TargetPort 8101 -OpenFirewall
```

---

## Verify Active Tablet Port

After portproxy is set, test active port `5000`:

```powershell
cd E:\emtac\prod

.\scripts\test_emtac_slot.ps1 -HostName 127.0.0.1 -Port 5000
.\scripts\test_emtac_slot.ps1 -HostName 172.19.194.129 -Port 5000
```

Expected:

```text
/health     -> 200 OK
/api-status -> 200 OK
```

---

## Freeze Requirements After the Slot Works

```powershell
cd E:\emtac\prod

.\scripts\freeze_emtac_slot.ps1 -Slot green
```

This creates:

```text
E:\emtac\prod\slots\green\requirements.green.lock.txt
E:\emtac\prod\slots\green\requirements.green.lock_<timestamp>.txt
```

The current lock file may stay in the slot, but old locks and dependency audit records should be moved to:

```text
E:\emtac\prod\shared\dependency_records
```

---

## Dependency Audit

Use the dependency audit script after the slot is working:

```powershell
cd E:\emtac\prod

.\scripts\audit_emtac_dependencies.ps1 -Slot green -OpenReport
```

This documents installed packages and wheel locations.

Important generated files:

```text
E:\emtac\prod\dependency_manifest.green.md
E:\emtac\prod\dependency_manifest.green.csv
E:\emtac\prod\dependency_manifest.green.json
E:\emtac\prod\missing_wheels.green.txt
E:\emtac\prod\wheel_inventory.csv
```

---

## pgvector Note

`pgvector==0.4.1` was manually copied into green because no matching pgvector wheel was found in:

```text
E:\wheels
```

The runtime import should resolve to the venv:

```text
E:\emtac\prod\slots\green\.venv\Lib\site-packages\pgvector
```

Verify:

```powershell
cd E:\emtac\prod

& "E:\emtac\prod\slots\green\.venv\Scripts\python.exe" -c "from pgvector.sqlalchemy import Vector; import pgvector; print(pgvector.__file__)"
```

Expected:

```text
E:\emtac\prod\slots\green\.venv\lib\site-packages\pgvector\__init__.py
```

The root folder below should not exist:

```text
E:\emtac\prod\slots\green\pgvector
```

For a clean future rebuild, add the `pgvector==0.4.1` wheel to the wheelhouse.

Until then, use:

```powershell
cd E:\emtac\prod

.\scripts\copy_pgvector_from_dev.ps1 -Slot green
```

---

## Firewall Notes

For LAN/tablet access, Windows firewall must allow the active port.

Common ports:

```text
5000 -> tablet/active
8101 -> blue
8102 -> green
```

Check listening ports:

```powershell
netstat -ano | findstr :5000
netstat -ano | findstr :8102
```

You want Waitress listening on:

```text
0.0.0.0:8102
```

not only:

```text
127.0.0.1:8102
```

---

## Portproxy Troubleshooting

If this is present:

```text
0.0.0.0  5000  ->  127.0.0.1  8102
```

but tablet access fails, check whether `8102` is actually listening.

This output means portproxy is trying to forward, but the slot is not listening:

```text
127.0.0.1:<random> -> 127.0.0.1:8102 SYN_SENT
```

Fix:

```powershell
cd E:\emtac\prod

.\scripts\start_emtac_slot.ps1 -Slot green -Port 8102 -HostAddress 0.0.0.0 -Force
```

Then retest:

```powershell
.\scripts\test_emtac_slot.ps1 -HostName 127.0.0.1 -Port 8102
.\scripts\test_emtac_slot.ps1 -HostName 127.0.0.1 -Port 5000
.\scripts\test_emtac_slot.ps1 -HostName 172.19.194.129 -Port 5000
```

---

## Foreground Startup for Debugging

If the background start script does not show the real error, run Waitress in the foreground:

```powershell
cd E:\emtac\prod\slots\green

$env:APP_ENV = "production"
$env:FLASK_ENV = "production"
$env:FLASK_DEBUG = "0"
$env:EMTAC_SLOT = "green"
$env:HOST = "0.0.0.0"
$env:PORT = "8102"
$env:EMTAC_SHARED_DIR = "E:\emtac\prod\shared"
$env:EMTAC_UPLOAD_FOLDER = "E:\emtac\prod\shared\uploads"
$env:EMTAC_LOG_DIR = "E:\emtac\prod\shared\logs"
$env:LOG_FILE = "E:\emtac\prod\shared\logs\green_app.log"

& ".\.venv\Scripts\python.exe" -m waitress --listen=0.0.0.0:8102 --threads=12 wsgi:app
```

Leave the foreground window open, then test from another PowerShell window.

---

## Common Problems

### Tablet cannot connect

Check:

1. Green or blue is running.
2. Port `5000` is forwarded to the active slot.
3. Windows firewall allows port `5000`.
4. Tablet is on the same network.
5. The active slot port is listening.

Test from the EMTAC machine:

```powershell
Invoke-WebRequest http://127.0.0.1:5000/health -UseBasicParsing
Invoke-WebRequest http://172.19.194.129:5000/health -UseBasicParsing
```

### Green health works on 127.0.0.1 but not LAN IP

Restart green using:

```text
0.0.0.0
```

not:

```text
127.0.0.1
```

### Missing package errors

Install the missing package into the slot venv using the offline wheelhouse, then freeze again.

Slot Python:

```text
E:\emtac\prod\slots\green\.venv\Scripts\python.exe
```

Wheelhouse:

```text
E:\wheels
```

Use:

```powershell
& "E:\emtac\prod\slots\green\.venv\Scripts\python.exe" -m pip install --no-index --find-links E:\wheels <package-name>
```

Then:

```powershell
cd E:\emtac\prod

.\scripts\freeze_emtac_slot.ps1 -Slot green
.\scripts\audit_emtac_dependencies.ps1 -Slot green
```

### Waitress task queue warnings

These warnings mean requests briefly backed up:

```text
waitress.queue - WARNING - Task queue depth is 1
waitress.queue - WARNING - Task queue depth is 2
```

Use more Waitress threads for tablet polling, help chat, and sync traffic:

```text
--threads=12
```

The production start script should include this by default.

### Unknown tablet offline events

Warnings such as:

```text
Unknown offline event_type received: heartbeat_failed
Unknown offline event_type received: offline_detected
```

are not fatal if the sync still reports:

```text
accepted=2 duplicates=0 failed=0
status=success
```

Later, add those event types to the tablet offline event normalization logic.

---

## Cleanup Rules

Production slots should not accumulate generated, test, or development files.

Use quarantine first:

```text
E:\emtac\prod\shared\cleanup_quarantine
```

Do not permanently delete until the slot has passed health checks.

After cleanup, test:

```powershell
cd E:\emtac\prod

.\scripts\test_emtac_slot.ps1 -HostName 127.0.0.1 -Port 8102
.\scripts\test_emtac_slot.ps1 -HostName 127.0.0.1 -Port 5000
.\scripts\test_emtac_slot.ps1 -HostName 172.19.194.129 -Port 5000
```

---

## Clean Green Status Checklist

A cleaned green slot should pass:

```powershell
cd E:\emtac\prod

.\scripts\test_emtac_slot.ps1 -HostName 127.0.0.1 -Port 8102
.\scripts\test_emtac_slot.ps1 -HostName 127.0.0.1 -Port 5000
.\scripts\test_emtac_slot.ps1 -HostName 172.19.194.129 -Port 5000
```

Expected:

```text
/health     -> 200 OK
/api-status -> 200 OK
```

Current known-good active tablet URL:

```text
http://172.19.194.129:5000/
```
