import subprocess
import os
import sys
import time
from datetime import datetime

# UPDATE THESE PATHS TO MATCH YOUR SETUP
BIN_DIR = r"E:\emtac\databases\postgresql\pgsql\bin"
DATA_DIR = r"E:\emtac\databases\postgresql\pgsql\data"


def run_pg_ctl(args, timeout=30):
    """Helper to run pg_ctl with given arguments and timeout."""
    pg_ctl_path = os.path.join(BIN_DIR, "pg_ctl.exe")
    cmd = [pg_ctl_path, "-D", DATA_DIR] + args

    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=BIN_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        return result
    except subprocess.TimeoutExpired:
        print(f"[WARNING] Command timed out after {timeout} seconds")
        return None
    except FileNotFoundError:
        print(f"[ERROR] pg_ctl.exe not found at: {pg_ctl_path}")
        print("Please check your BIN_DIR path in the script.")
        return None
    except PermissionError:
        print(f"[ERROR] Permission denied. Try running as Administrator or check folder permissions.")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return None


def start_postgres():
    """Start PostgreSQL server."""
    print("[INFO] Starting PostgreSQL server...")

    # Check if already running first
    status_result = run_pg_ctl(["status"])
    if status_result and "server is running" in status_result.stdout:
        print("[SUCCESS] PostgreSQL is already running!")
        return

    # Try to start with log file and shorter timeout
    log_file = os.path.join(DATA_DIR, "server.log")
    print("[INFO] Attempting to start PostgreSQL (this may take a few seconds)...")

    result = run_pg_ctl(["-l", log_file, "start", "-w"], timeout=20)  # Added -w flag and shorter timeout

    if result is None:
        print("[ERROR] Start command timed out or failed")
        try_manual_start()
        return

    print(f"[INFO] Return code: {result.returncode}")

    if result.stdout.strip():
        print(f"[OUTPUT] {result.stdout.strip()}")

    if result.stderr.strip():
        print(f"[ERROR] {result.stderr.strip()}")

    if result.returncode == 0:
        print("[SUCCESS] PostgreSQL server started!")
        # Brief pause then verify
        print("[INFO] Verifying server status...")
        time.sleep(1)
        if verify_status():
            print("[INFO] Server startup completed successfully")
        else:
            print("[WARNING] Server may still be starting up")
    else:
        print("[ERROR] Failed to start PostgreSQL server")
        show_recent_logs(log_file)
        try_manual_start()

    print("[INFO] Start operation completed - returning to menu")


def stop_postgres():
    """Stop PostgreSQL server."""
    print("[INFO] Stopping PostgreSQL server...")
    result = run_pg_ctl(["stop"])

    if result is None:
        print("[ERROR] Failed to stop PostgreSQL")
        return

    if result.returncode == 0:
        print("[SUCCESS] PostgreSQL server stopped.")
    else:
        print("[ERROR] Error stopping PostgreSQL server:")
        if result.stderr.strip():
            print(f"[ERROR] {result.stderr.strip()}")


def status_postgres():
    """Check PostgreSQL server status."""
    print("[INFO] Checking PostgreSQL server status...")
    result = run_pg_ctl(["status"])

    if result is None:
        print("[ERROR] Failed to check status")
        return

    if "server is running" in result.stdout:
        print("[SUCCESS] PostgreSQL server is RUNNING")
        lines = result.stdout.split('\n')
        for line in lines:
            if "PID" in line:
                print(f"[INFO] {line.strip()}")
    elif "no server running" in result.stdout:
        print("[INFO] PostgreSQL server is NOT running")
    else:
        print("[WARNING] Unable to determine server status")
        if result.stdout.strip():
            print(f"[OUTPUT] {result.stdout.strip()}")


def verify_status():
    """Quick status verification."""
    result = run_pg_ctl(["status"])
    if result and "server is running" in result.stdout:
        print("[SUCCESS] Status verified: Server is running")
        return True
    else:
        print("[WARNING] Server may not be fully started")
        return False


def show_recent_logs(log_file=None, lines=10):
    """Show recent entries from PostgreSQL log file."""
    if log_file is None:
        log_file = os.path.join(DATA_DIR, "server.log")

    try:
        if os.path.exists(log_file):
            print(f"\n[INFO] Last {lines} lines from {log_file}:")
            print("-" * 50)
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                for line in recent_lines:
                    print(line.rstrip())
            print("-" * 50)
        else:
            print(f"[WARNING] Log file not found: {log_file}")
    except Exception as e:
        print(f"[ERROR] Could not read log file: {e}")


def check_paths():
    """Verify that required paths exist."""
    print("[INFO] Checking configuration...")

    if not os.path.exists(BIN_DIR):
        print(f"[ERROR] BIN_DIR not found: {BIN_DIR}")
        print("[FIX] Update BIN_DIR in the script to point to your PostgreSQL bin folder")
        return False

    pg_ctl_path = os.path.join(BIN_DIR, "pg_ctl.exe")
    if not os.path.exists(pg_ctl_path):
        print(f"[ERROR] pg_ctl.exe not found: {pg_ctl_path}")
        return False

    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] DATA_DIR not found: {DATA_DIR}")
        print("[FIX] Update DATA_DIR in the script or initialize database")
        return False

    config_file = os.path.join(DATA_DIR, "postgresql.conf")
    if not os.path.exists(config_file):
        print(f"[ERROR] PostgreSQL config not found: {config_file}")
        print("[FIX] Initialize database with initdb")
        return False

    print("[SUCCESS] All paths and files look good!")
    return True


def try_manual_start():
    """Provide manual start commands when automatic start fails."""
    print("\n[MANUAL] Try these commands manually:")
    print("=" * 40)
    print("1. Open Command Prompt (cmd)")
    print(f"2. cd \"{BIN_DIR}\"")
    print(f"3. pg_ctl.exe -D \"{DATA_DIR}\" start")
    print("\nOR with PowerShell:")
    print(f"1. cd \"{BIN_DIR}\"")
    print(f"2. .\\pg_ctl.exe -D \"{DATA_DIR}\" start")


def quick_diagnosis():
    """Run quick diagnostic checks."""
    print("\n[DIAGNOSIS] Running checks...")
    print("-" * 30)

    issues = []

    # Check paths
    if not os.path.exists(BIN_DIR):
        issues.append(f"BIN_DIR not found: {BIN_DIR}")
    else:
        print(f"[OK] BIN_DIR exists: {BIN_DIR}")

    pg_ctl_path = os.path.join(BIN_DIR, "pg_ctl.exe")
    if not os.path.exists(pg_ctl_path):
        issues.append(f"pg_ctl.exe not found: {pg_ctl_path}")
    else:
        print(f"[OK] pg_ctl.exe found")

    if not os.path.exists(DATA_DIR):
        issues.append(f"DATA_DIR not found: {DATA_DIR}")
    else:
        print(f"[OK] DATA_DIR exists: {DATA_DIR}")

    # Check database initialization
    config_file = os.path.join(DATA_DIR, "postgresql.conf")
    if not os.path.exists(config_file):
        issues.append("Database not initialized (postgresql.conf missing)")
    else:
        print("[OK] Database appears initialized")

    # Check PostgreSQL version
    try:
        version_file = os.path.join(DATA_DIR, "PG_VERSION")
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                version = f.read().strip()
            print(f"[OK] PostgreSQL version: {version}")
        else:
            issues.append("PG_VERSION file not found")
    except:
        print("[WARNING] Could not read PostgreSQL version")

    # Summary
    if issues:
        print(f"\n[ISSUES] Found {len(issues)} issue(s):")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("\n[SUCCESS] No obvious issues detected!")


def show_connection_info():
    """Display connection information."""
    print("\n[CONNECTION] PostgreSQL Connection Info:")
    print("   Host: localhost")
    print("   Port: 5432")
    print("   Default Database: postgres")
    print("   User: postgres")
    print(f"   Test Connection: psql -U postgres -h localhost")


def verify_start_nonblocking():
    """Verify start with immediate return - for debugging hanging issues."""
    print("[Verify ] Verifying non-blocking start...")

    # Check current status first
    print("[VERIFY] Step 1: Checking current status...")
    status_result = run_pg_ctl(["status"])
    if status_result and "server is running" in status_result.stdout:
        print("[VERIFY] PostgreSQL is already running!")
        return

    print("[VERIFY] Step 2: Attempting start with 5-second timeout...")
    log_file = os.path.join(DATA_DIR, "server.log")

    # Very short timeout to see if it returns quickly
    result = run_pg_ctl(["-l", log_file, "start", "-w"], timeout=5)

    if result is None:
        print("[VERIFY] Command timed out after 5 seconds - this indicates hanging")
        print("[VERIFY] Try manual start: pg_ctl.exe -D DATA_DIR start")
    else:
        print(f"[VERIFY] Command returned with code: {result.returncode}")
        print(f"[VERIFY] Output: {result.stdout.strip()}")
        if result.stderr.strip():
            print(f"[VERIFY] Errors: {result.stderr.strip()}")

    print("[VERIFY] Test completed")


def fix_common_issues():
    """Provide solutions for common issues."""
    print("\n[FIXES] Common Issue Solutions:")
    print("=" * 30)

    print("\n1. Database Not Initialized:")
    print(f"   cd \"{BIN_DIR}\"")
    print(f"   initdb.exe -D \"{DATA_DIR}\" -U postgres")

    print("\n2. Permission Issues:")
    print("   - Ensure your user has read/write access to DATA_DIR")
    print("   - Try running Command Prompt as Administrator")
    print("   - Check Windows folder permissions")

    print("\n3. Port Already in Use:")
    print("   netstat -an | findstr :5432")
    print("   - Stop other PostgreSQL instances")
    print("   - Change port in postgresql.conf")

    print("\n4. Path Issues:")
    print("   - Update BIN_DIR and DATA_DIR in this script")
    print("   - Use full paths, not relative paths")


def main():
    """Main menu loop."""
    print("PostgreSQL Server Control Panel (No Admin Required)")
    print("=" * 50)

    # Check paths on startup
    if not check_paths():
        print("\n[ERROR] Configuration issues detected.")
        print("[FIX] Update the BIN_DIR and DATA_DIR paths at the top of this script")
        input("Press Enter to exit...")
        return

    while True:
        print(f"\n[TIME] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("PostgreSQL Control Menu")
        print("=" * 22)
        print("1. Start server")
        print("2. Stop server")
        print("3. Check status")
        print("4. Show recent logs")
        print("5. Connection info")
        print("6. Quick diagnosis")
        print("7. Fix common issues")
        print("8. Exit")

        choice = input("\nChoose option [1-8]: ").strip()

        if choice == '1':
            start_postgres()
        elif choice == '2':
            stop_postgres()
        elif choice == '3':
            status_postgres()
        elif choice == '4':
            show_recent_logs()
        elif choice == '5':
            show_connection_info()
        elif choice == '6':
            quick_diagnosis()
        elif choice == '7':
            fix_common_issues()
        elif choice == '8':
            print("Goodbye!")
            break
        else:
            print("[ERROR] Invalid option. Please enter 1-8.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()