# services/postgres_server/postgres_server.py

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from typing import Optional, Dict, Any, List

from .config import PostgresServerConfig
from .logger import get_logger, mask_sensitive_config


logger = get_logger()


class PostgresServerManager:
    """
    Service-style manager for starting, stopping, checking, and diagnosing
    a local PostgreSQL instance controlled via pg_ctl.
    """

    def __init__(self, config: Optional[PostgresServerConfig] = None):
        self.config = config or PostgresServerConfig()

        safe_config = mask_sensitive_config(asdict(self.config))

        logger.info(
            "PostgresServerManager initialized | config=%s",
            safe_config,
        )

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------
    def _creation_flags(self) -> int:
        if os.name == "nt":
            return subprocess.CREATE_NO_WINDOW
        return 0

    def _run_pg_ctl(
        self,
        args: List[str],
        timeout: int = 30,
    ) -> Optional[subprocess.CompletedProcess]:
        cmd = [self.config.pg_ctl_path, "-D", self.config.data_dir] + args

        try:
            logger.info("Running command: %s", " ".join(cmd))
            result = subprocess.run(
                cmd,
                cwd=self.config.bin_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                creationflags=self._creation_flags(),
            )

            if result.stdout.strip():
                logger.info("pg_ctl stdout: %s", result.stdout.strip())

            if result.stderr.strip():
                logger.warning("pg_ctl stderr: %s", result.stderr.strip())

            return result

        except subprocess.TimeoutExpired:
            logger.error("Command timed out after %s seconds", timeout)
            return None
        except FileNotFoundError:
            logger.exception("pg_ctl.exe not found: %s", self.config.pg_ctl_path)
            return None
        except PermissionError:
            logger.exception("Permission denied while running pg_ctl")
            return None
        except Exception:
            logger.exception("Unexpected error while running pg_ctl")
            return None

    def _status_result(self) -> Optional[subprocess.CompletedProcess]:
        return self._run_pg_ctl(["status"], timeout=10)

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------
    def check_paths(self) -> Dict[str, Any]:
        issues = []

        if not os.path.exists(self.config.bin_dir):
            issues.append(f"POSTGRES_BIN_DIR not found: {self.config.bin_dir}")

        if not os.path.exists(self.config.pg_ctl_path):
            issues.append(f"pg_ctl.exe not found: {self.config.pg_ctl_path}")

        if not os.path.exists(self.config.data_dir):
            issues.append(f"POSTGRES_DATA_DIR not found: {self.config.data_dir}")

        if os.path.exists(self.config.data_dir) and not os.path.exists(self.config.config_file):
            issues.append(f"postgresql.conf not found: {self.config.config_file}")

        ok = len(issues) == 0

        if ok:
            logger.info("PostgreSQL path validation successful")
        else:
            logger.error("PostgreSQL path validation failed | issues=%s", issues)

        return {
            "ok": ok,
            "issues": issues,
            "bin_dir": self.config.bin_dir,
            "data_dir": self.config.data_dir,
            "pg_ctl_path": self.config.pg_ctl_path,
            "config_file": self.config.config_file,
        }

    # ---------------------------------------------------------
    # Status
    # ---------------------------------------------------------
    def is_running(self) -> bool:
        result = self._status_result()
        return bool(result and "server is running" in result.stdout.lower())

    def get_status(self) -> Dict[str, Any]:
        result = self._status_result()

        if result is None:
            return {
                "ok": False,
                "running": False,
                "message": "Failed to check PostgreSQL status",
            }

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if "server is running" in stdout.lower():
            return {
                "ok": True,
                "running": True,
                "message": "PostgreSQL server is running",
                "stdout": stdout,
                "stderr": stderr,
                "returncode": result.returncode,
            }

        if "no server running" in stdout.lower():
            return {
                "ok": True,
                "running": False,
                "message": "PostgreSQL server is not running",
                "stdout": stdout,
                "stderr": stderr,
                "returncode": result.returncode,
            }

        return {
            "ok": False,
            "running": False,
            "message": "Unable to determine PostgreSQL status",
            "stdout": stdout,
            "stderr": stderr,
            "returncode": result.returncode,
        }

    def verify_status(self) -> bool:
        running = self.is_running()
        if running:
            logger.info("Status verified: PostgreSQL is running")
        else:
            logger.warning("Status verification failed: PostgreSQL is not running")
        return running

    # ---------------------------------------------------------
    # Start / stop / ensure / restart
    # ---------------------------------------------------------
    def start(self, timeout: int = 20) -> Dict[str, Any]:
        logger.info("Starting PostgreSQL server...")

        if self.is_running():
            logger.info("PostgreSQL is already running")
            return {
                "ok": True,
                "changed": False,
                "running": True,
                "message": "PostgreSQL is already running",
            }

        result = self._run_pg_ctl(
            ["-l", self.config.log_file, "start", "-w"],
            timeout=timeout,
        )

        if result is None:
            logger.error("PostgreSQL start command failed or timed out")
            return {
                "ok": False,
                "changed": False,
                "running": False,
                "message": "PostgreSQL start command failed or timed out",
                "manual_start": self.get_manual_start_commands(),
            }

        time.sleep(1)
        running = self.verify_status()

        if result.returncode == 0 and running:
            logger.info("PostgreSQL server started successfully")
            return {
                "ok": True,
                "changed": True,
                "running": True,
                "message": "PostgreSQL server started successfully",
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode,
            }

        logger.error("Failed to start PostgreSQL server")
        return {
            "ok": False,
            "changed": False,
            "running": running,
            "message": "Failed to start PostgreSQL server",
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
            "recent_logs": self.get_recent_logs(),
            "manual_start": self.get_manual_start_commands(),
        }

    def stop(self, timeout: int = 45) -> Dict[str, Any]:
        logger.info("Stopping PostgreSQL server...")

        if not self.is_running():
            logger.info("PostgreSQL is already stopped")
            return {
                "ok": True,
                "changed": False,
                "running": False,
                "message": "PostgreSQL is already stopped",
            }

        result = self._run_pg_ctl(["stop"], timeout=timeout)

        if result is None:
            logger.error("Failed to stop PostgreSQL server")
            return {
                "ok": False,
                "changed": False,
                "running": True,
                "message": "Failed to stop PostgreSQL server",
            }

        time.sleep(1)
        running = self.is_running()

        if result.returncode == 0 and not running:
            logger.info("PostgreSQL server stopped successfully")
            return {
                "ok": True,
                "changed": True,
                "running": False,
                "message": "PostgreSQL server stopped successfully",
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode,
            }

        logger.error("PostgreSQL stop command completed but server still appears to be running")
        return {
            "ok": False,
            "changed": False,
            "running": running,
            "message": "Stop command completed but server still appears to be running",
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }

    def ensure_running(self, timeout: int = 20) -> Dict[str, Any]:
        """
        Ensure PostgreSQL is running.
        Returns a standard result dict whether it was already running or had to be started.
        """
        logger.info("Ensuring PostgreSQL server is running...")

        if self.is_running():
            logger.info("PostgreSQL already running")
            return {
                "ok": True,
                "changed": False,
                "running": True,
                "message": "PostgreSQL already running",
            }

        return self.start(timeout=timeout)

    def ensure_stopped(self, timeout: int = 45) -> Dict[str, Any]:
        """
        Ensure PostgreSQL is stopped.
        Returns a standard result dict whether it was already stopped or had to be stopped.
        """
        logger.info("Ensuring PostgreSQL server is stopped...")

        if not self.is_running():
            logger.info("PostgreSQL already stopped")
            return {
                "ok": True,
                "changed": False,
                "running": False,
                "message": "PostgreSQL already stopped",
            }

        return self.stop(timeout=timeout)

    def restart(self, timeout: int = 45) -> Dict[str, Any]:
        logger.info("Restarting PostgreSQL server...")

        stop_result = self.ensure_stopped(timeout=timeout)
        start_result = self.ensure_running(timeout=timeout)

        return {
            "ok": stop_result.get("ok") and start_result.get("ok"),
            "message": "Restart attempted",
            "stop_result": stop_result,
            "start_result": start_result,
        }

    # ---------------------------------------------------------
    # Logs / diagnosis
    # ---------------------------------------------------------
    def get_recent_logs(self, lines: int = 10) -> List[str]:
        log_file = self.config.log_file

        try:
            if not os.path.exists(log_file):
                logger.warning("Log file not found: %s", log_file)
                return []

            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                all_lines = [line.rstrip() for line in f.readlines()]

            recent = all_lines[-lines:] if len(all_lines) > lines else all_lines
            logger.info("Loaded %s recent log lines from %s", len(recent), log_file)
            return recent

        except Exception:
            logger.exception("Could not read PostgreSQL log file")
            return []

    def quick_diagnosis(self) -> Dict[str, Any]:
        logger.info("Running PostgreSQL quick diagnosis...")

        issues = []
        checks = {}

        checks["bin_dir_exists"] = os.path.exists(self.config.bin_dir)
        if not checks["bin_dir_exists"]:
            issues.append(f"POSTGRES_BIN_DIR not found: {self.config.bin_dir}")

        checks["pg_ctl_exists"] = os.path.exists(self.config.pg_ctl_path)
        if not checks["pg_ctl_exists"]:
            issues.append(f"pg_ctl.exe not found: {self.config.pg_ctl_path}")

        checks["data_dir_exists"] = os.path.exists(self.config.data_dir)
        if not checks["data_dir_exists"]:
            issues.append(f"POSTGRES_DATA_DIR not found: {self.config.data_dir}")

        checks["config_exists"] = os.path.exists(self.config.config_file)
        if not checks["config_exists"]:
            issues.append(f"postgresql.conf not found: {self.config.config_file}")

        version = None
        if os.path.exists(self.config.version_file):
            try:
                with open(self.config.version_file, "r", encoding="utf-8", errors="ignore") as f:
                    version = f.read().strip()
            except Exception:
                logger.exception("Could not read PG_VERSION file")

        status = self.get_status()

        result = {
            "ok": len(issues) == 0,
            "issues": issues,
            "checks": checks,
            "version": version,
            "status": status,
            "host": self.config.host,
            "port": self.config.port,
            "user": self.config.user,
            "database": self.config.database,
            "database_url": self.config.database_url,
        }

        logger.info("Diagnosis complete | result=%s", mask_sensitive_config(result))
        return result

    # ---------------------------------------------------------
    # Helper text info
    # ---------------------------------------------------------
    def get_manual_start_commands(self) -> Dict[str, str]:
        return {
            "cmd": f'cd /d "{self.config.bin_dir}" && pg_ctl.exe -D "{self.config.data_dir}" start',
            "powershell": f'cd "{self.config.bin_dir}"; .\\pg_ctl.exe -D "{self.config.data_dir}" start',
        }

    def get_connection_info(self) -> Dict[str, Any]:
        return {
            "host": self.config.host,
            "port": self.config.port,
            "database": self.config.database,
            "user": self.config.user,
            "database_url": self.config.database_url,
            "psql_test_command": (
                f'psql -U {self.config.user} -h {self.config.host} '
                f'-p {self.config.port} -d {self.config.database}'
            ),
        }

    # ---------------------------------------------------------
    # Optional simple console menu
    # ---------------------------------------------------------
    def run_console_menu(self) -> None:
        print("PostgreSQL Server Control Panel")
        print("=" * 40)

        path_check = self.check_paths()
        if not path_check["ok"]:
            print("\nConfiguration issues detected:")
            for issue in path_check["issues"]:
                print(f" - {issue}")
            return

        while True:
            print(f"\n[TIME] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("1. Start server")
            print("2. Stop server")
            print("3. Check status")
            print("4. Show recent logs")
            print("5. Connection info")
            print("6. Quick diagnosis")
            print("7. Restart")
            print("8. Exit")

            choice = input("\nChoose option [1-8]: ").strip()

            if choice == "1":
                print(self.ensure_running())
            elif choice == "2":
                print(self.ensure_stopped())
            elif choice == "3":
                print(self.get_status())
            elif choice == "4":
                for line in self.get_recent_logs():
                    print(line)
            elif choice == "5":
                print(mask_sensitive_config(self.get_connection_info()))
            elif choice == "6":
                print(mask_sensitive_config(self.quick_diagnosis()))
            elif choice == "7":
                print(self.restart())
            elif choice == "8":
                print("Goodbye")
                break
            else:
                print("Invalid option")