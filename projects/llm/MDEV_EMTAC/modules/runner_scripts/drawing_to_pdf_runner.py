import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from coordinators.drawing_to_pdf_coordinator import DrawingToPDFCoordinator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DrawingToPDF runner script (single file or folder).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    single_parser = subparsers.add_parser("single", help="Convert one file to PDF.")
    single_parser.add_argument("input_file", help="Source drawing file path.")
    single_parser.add_argument("output_file", nargs="?", help="Optional output PDF path.")
    single_parser.add_argument("--quality", type=int, choices=[1, 2, 3], default=2)
    single_parser.add_argument("--timeout", type=int, default=180)
    single_parser.add_argument("--visible", action="store_true")
    single_parser.add_argument(
        "--background-mode",
        choices=["white", "black", "auto"],
        default="white",
        help="Background policy: white=leave as is, black=force black, auto=detect per output.",
    )
    single_parser.add_argument("--black-background", action="store_true", help=argparse.SUPPRESS)

    folder_parser = subparsers.add_parser("folder", help="Convert a folder of drawings to PDF.")
    folder_parser.add_argument("input_folder", help="Source folder.")
    folder_parser.add_argument("output_folder", help="Output folder.")
    folder_parser.add_argument("--non-recursive", action="store_true")
    folder_parser.add_argument("--quality", type=int, choices=[1, 2, 3], default=2)
    folder_parser.add_argument("--timeout", type=int, default=180)
    folder_parser.add_argument("--visible", action="store_true")
    folder_parser.add_argument("--fail-fast", action="store_true", help="Stop at first failure.")
    folder_parser.add_argument("--max-files", type=int)
    folder_parser.add_argument("--all-files", action="store_true")
    folder_parser.add_argument("--dry-run", action="store_true")
    folder_parser.add_argument("--sldprocmon-check-interval", type=int, default=5)
    folder_parser.add_argument("--cpu-target-percent", type=float, default=85.0)
    folder_parser.add_argument("--cpu-sample-seconds", type=float, default=0.7)
    folder_parser.add_argument("--cpu-throttle-max-wait", type=float, default=20.0)
    folder_parser.add_argument(
        "--background-mode",
        choices=["white", "black", "auto"],
        default="white",
        help="Background policy: white=leave as is, black=force black, auto=detect per output.",
    )
    folder_parser.add_argument("--black-background", action="store_true", help=argparse.SUPPRESS)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    coordinator = DrawingToPDFCoordinator()
    background_mode = "black" if args.black_background else args.background_mode

    try:
        if args.command == "single":
            result = coordinator.run_single(
                source_file=args.input_file,
                output_file=args.output_file,
                quality=args.quality,
                timeout=args.timeout,
                visible=args.visible,
                background_mode=background_mode,
            )
        else:
            result = coordinator.run_folder(
                input_folder=args.input_folder,
                output_folder=args.output_folder,
                recursive=not args.non_recursive,
                quality=args.quality,
                timeout=args.timeout,
                visible=args.visible,
                keep_going=not args.fail_fast,
                max_files=args.max_files,
                all_files=args.all_files,
                dry_run=args.dry_run,
                sldprocmon_check_interval=args.sldprocmon_check_interval,
                cpu_target_percent=args.cpu_target_percent,
                cpu_sample_seconds=args.cpu_sample_seconds,
                cpu_throttle_max_wait=args.cpu_throttle_max_wait,
                background_mode=background_mode,
            )
    except Exception as exc:
        error_payload = {"success": False, "error": str(exc)}
        print(json.dumps(error_payload, indent=2))
        traceback.print_exc()
        return 1

    print(json.dumps(result, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
