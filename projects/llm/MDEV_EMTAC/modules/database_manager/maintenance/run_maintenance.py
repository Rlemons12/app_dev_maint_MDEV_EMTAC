# C:\Users\10169062\Desktop\AU_IndusMaintdb\modules\database_manager\maintenance\run_maintenance.py
# Enhanced CLI script to run optimized database maintenance functions

import sys
import os
import subprocess
import argparse
import time
from datetime import datetime


def get_script_path():
    """Get the path to the optimized maintenance script."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, 'optimized_db_maintenance.py')


def print_banner():
    """Print a nice banner for the maintenance tool."""
    print("=" * 60)
    print("OPTIMIZED DATABASE MAINTENANCE TOOL")
    print("=" * 60)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def run_maintenance_task(task, report_dir=None, export_report=True):
    """
    Run a specific maintenance task using the optimized script.

    Args:
        task: The maintenance task to run
        report_dir: Directory to save reports
        export_report: Whether to export reports

    Returns:
        Exit code (0 for success)
    """
    python_exe = sys.executable
    maintenance_script = get_script_path()

    if not os.path.exists(maintenance_script):
        print(f"Error: Optimized maintenance script not found at {maintenance_script}")
        print("💡 Make sure optimized_db_maintenance.py exists in the same directory")
        return 1

    # Build the command
    cmd = [python_exe, "-m", "modules.database_manager.maintenance.optimized_db_maintenance", task]


    # Add report options
    if report_dir:
        cmd.extend(['--report-dir', report_dir])

    if export_report:
        cmd.append('--export-report')
    else:
        cmd.append('--no-export-report')

    # Show what we're running
    print(f"🔧 Running: {task}")
    print(f"📁 Report directory: {report_dir or 'default (db_maint_logs)'}")
    print(f"Export reports: {'Yes' if export_report else 'No'}")
    print()

    # Execute the command
    start_time = time.time()
    print(f"▶ Executing: {' '.join(cmd)}")
    print("-" * 50)

    result = subprocess.run(cmd)

    print("-" * 50)
    duration = time.time() - start_time

    if result.returncode == 0:
        print(f"Task '{task}' completed successfully in {duration:.2f} seconds!")
    else:
        print(f"Task '{task}' failed with exit code {result.returncode}")
        print(f"⏱ Duration: {duration:.2f} seconds")

    return result.returncode


def main():
    """Main entry point for the enhanced maintenance CLI"""
    print_banner()

    parser = argparse.ArgumentParser(
        description='Run optimized database maintenance tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Tasks:
  associate-images     🖼 Associate parts with matching images (FAST)
  associate-drawings   📋 Associate drawings with matching parts (FAST) 
  run-all             Run ALL maintenance operations (FASTEST)

Examples:
  python run_maintenance.py --task associate-images
  python run_maintenance.py --task run-all --report-dir ./reports
  python run_maintenance.py --task associate-drawings --no-report
        """
    )

    # Add arguments with better descriptions
    parser.add_argument(
        '--task',
        choices=['associate-images', 'associate-drawings', 'validate-embeddings','add-image-embeddings', 'run-all'],
        default='run-all',
        help='Maintenance task to run (default: run-all - runs everything!)'
    )

    parser.add_argument(
        '--report-dir',
        type=str,
        help='Directory to save reports (default: db_maint_logs)',
        default=None
    )

    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Do not generate report files'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run in quick mode (minimal output)'
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.quick:
        print(f"Selected task: {args.task}")
        if args.task == 'run-all':
            print("   This will run ALL optimized maintenance operations!")
        print()

    # Map task names to actual CLI commands
    task_mapping = {
        'associate-images': 'associate-images-fast',
        'associate-drawings': 'associate-drawings-fast',
        'validate-embeddings': 'validate-embeddings',
        'add-image-embeddings': 'add-image-embeddings',
        'run-all': 'run-all-fast'
    }

    actual_task = task_mapping[args.task]
    export_report = not args.no_report

    try:
        # Run the maintenance task
        result = run_maintenance_task(
            task=actual_task,
            report_dir=args.report_dir,
            export_report=export_report
        )

        if result == 0:
            print()
            print("🎉 MAINTENANCE COMPLETED SUCCESSFULLY!")
            print("=" * 60)

            if export_report:
                report_dir = args.report_dir or "db_maint_logs"
                print(f"Reports saved to: {os.path.abspath(report_dir)}")

        else:
            print()
            print("⚠ MAINTENANCE COMPLETED WITH ERRORS")
            print("=" * 60)
            print("💡 Check the output above for error details")

        return result

    except KeyboardInterrupt:
        print("\n🛑 Maintenance interrupted by user")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())