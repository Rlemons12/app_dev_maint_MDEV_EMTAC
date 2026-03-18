from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote

RAW_REQUIREMENTS = r"""
accelerate @ file:///E:/wheels/gpu_service_wheels/accelerate-1.12.0-py3-none-any.whl
annotated-doc @ file:///E:/wheels/gpu_service_wheels/annotated_doc-0.0.4-py3-none-any.whl
annotated-types @ file:///E:/wheels/gpu_service_wheels/annotated_types-0.7.0-py3-none-any.whl
anyio @ file:///E:/wheels/gpu_service_wheels/anyio-4.12.0-py3-none-any.whl
bidict==0.23.1
blinker==1.9.0
certifi==2025.11.12
charset-normalizer==3.4.4
click @ file:///E:/wheels/gpu_service_wheels/click-8.3.1-py3-none-any.whl
colorama==0.4.6
exceptiongroup==1.3.1
fastapi @ file:///E:/wheels/gpu_service_wheels/fastapi-0.124.4-py3-none-any.whl
filelock @ file:///E:/wheels/gpu_service_wheels/filelock-3.20.1-py3-none-any.whl
Flask==3.1.3
Flask-SocketIO==5.6.1
fsspec @ file:///E:/wheels/gpu_service_wheels/fsspec-2025.12.0-py3-none-any.whl
h11 @ file:///E:/wheels/gpu_service_wheels/h11-0.16.0-py3-none-any.whl
huggingface-hub==0.36.0
idna==3.11
iniconfig==2.3.0
itsdangerous==2.2.0
Jinja2==3.1.6
joblib @ file:///E:/wheels/gpu_service_wheels/joblib-1.5.3-py3-none-any.whl
MarkupSafe @ file:///E:/wheels/gpu_service_wheels/markupsafe-3.0.3-cp310-cp310-win_amd64.whl
mpmath==1.3.0
networkx @ file:///E:/wheels/gpu_service_wheels/networkx-3.4.2-py3-none-any.whl
numpy @ file:///E:/wheels/gpu_service_wheels/numpy-2.2.6-cp310-cp310-win_amd64.whl
nvidia-ml-py==13.590.48
packaging==25.0
pillow==11.3.0
pluggy==1.6.0
protobuf==3.20.3
psutil==7.1.3
pydantic==2.12.5
pydantic_core==2.41.5
Pygments==2.19.2
PyMuPDF @ file:///E:/wheels/ui_emtac/PyMuPDF-1.24.1-cp310-none-win_amd64.whl
PyMuPDFb @ file:///E:/wheels/ui_emtac/PyMuPDFb-1.24.1-py3-none-win_amd64.whl
pynvml==13.0.1
pytest==9.0.2
python-dotenv @ file:///E:/wheels/gpu_service_wheels/python_dotenv-1.2.1-py3-none-any.whl
python-engineio==4.13.1
python-multipart==0.0.22
python-socketio==5.16.1
PyYAML==6.0.3
regex @ file:///E:/wheels/gpu_service_wheels/regex-2025.11.3-cp310-cp310-win_amd64.whl
requests==2.32.5
safetensors==0.7.0
scikit-learn @ file:///E:/wheels/gpu_service_wheels/scikit_learn-1.7.2-cp310-cp310-win_amd64.whl
scipy==1.15.3
sentence-transformers @ file:///E:/wheels/gpu_service_wheels/sentence_transformers-5.2.0-py3-none-any.whl
sentencepiece==0.1.99
simple-websocket==1.1.0
starlette==0.50.0
sympy==1.14.0
threadpoolctl==3.6.0
tokenizers==0.22.1
tomli==2.4.0
torch @ file:///E:/wheels/torch_cu118_py310_wheels/torch-2.7.1%2Bcu118-cp310-cp310-win_amd64.whl
torchaudio @ file:///E:/wheels/torch_cu118_py310_wheels/torchaudio-2.7.1%2Bcu118-cp310-cp310-win_amd64.whl
torchvision @ file:///E:/wheels/torch_cu118_py310_wheels/torchvision-0.22.1%2Bcu118-cp310-cp310-win_amd64.whl
tqdm==4.67.1
transformers==4.57.3
typing-inspection==0.4.2
typing_extensions==4.15.0
urllib3==2.6.2
uvicorn @ file:///E:/wheels/gpu_service_wheels/uvicorn-0.38.0-py3-none-any.whl
Werkzeug==3.1.6
wsproto==1.3.2
""".strip()


@dataclass
class RequirementEntry:
    raw_line: str
    package_name: str
    normalized_name: str
    version: Optional[str] = None
    source_url: Optional[str] = None
    source_file_path: Optional[str] = None
    original_type: str = "pinned"  # pinned | file_url


@dataclass
class WheelInfo:
    path: Path
    file_name: str
    normalized_name: str
    version: str


@dataclass
class MatchResult:
    requirement: RequirementEntry
    found: bool
    match_strategy: str  # exact_url_path | recursive_exact_version | recursive_name_only | missing
    matched_wheel: Optional[str] = None
    matched_version: Optional[str] = None
    exact_version_match: bool = False
    candidate_count: int = 0
    candidates: Optional[List[str]] = None
    note: Optional[str] = None


def normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def file_url_to_windows_path(file_url: str) -> str:
    """
    Convert:
      file:///E:/path/to/file.whl
    into:
      E:\\path\\to\\file.whl

    Also URL-decodes escaped characters such as:
      %2B -> +
      %20 -> space
    """
    value = file_url.strip()

    if value.lower().startswith("file:///"):
        value = value[8:]
    elif value.lower().startswith("file://"):
        value = value[7:]

    value = unquote(value)
    value = value.replace("/", "\\")
    return value


def parse_requirement_line(line: str) -> Optional[RequirementEntry]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    if " @ " in stripped:
        pkg, src = stripped.split(" @ ", 1)
        pkg = pkg.strip()
        src = src.strip()
        return RequirementEntry(
            raw_line=stripped,
            package_name=pkg,
            normalized_name=normalize_package_name(pkg),
            version=extract_version_from_url_or_filename(src),
            source_url=src,
            source_file_path=file_url_to_windows_path(src) if src.lower().startswith("file:") else None,
            original_type="file_url",
        )

    if "==" in stripped:
        pkg, ver = stripped.split("==", 1)
        pkg = pkg.strip()
        ver = ver.strip()
        return RequirementEntry(
            raw_line=stripped,
            package_name=pkg,
            normalized_name=normalize_package_name(pkg),
            version=ver,
            original_type="pinned",
        )

    return RequirementEntry(
        raw_line=stripped,
        package_name=stripped,
        normalized_name=normalize_package_name(stripped),
        version=None,
        original_type="pinned",
    )


def extract_version_from_url_or_filename(url_or_path: str) -> Optional[str]:
    normalized = unquote(url_or_path.replace("\\", "/"))
    filename = normalized.split("/")[-1]
    parsed = parse_wheel_filename(filename)
    return parsed.version if parsed else None


def parse_wheel_filename(filename: str) -> Optional[WheelInfo]:
    """
    Parses wheel filename enough to get:
    - distribution name
    - version

    Assumes standard wheel naming:
      {distribution}-{version}(-{build})?-{python tag}-{abi tag}-{platform tag}.whl
    """
    if not filename.lower().endswith(".whl"):
        return None

    stem = filename[:-4]
    parts = stem.split("-")

    if len(parts) < 5:
        return None

    py_tag_index = None
    for i, part in enumerate(parts):
        if re.match(r"^(py\d|cp\d|pp\d|ip\d)", part):
            py_tag_index = i
            break

    if py_tag_index is None or py_tag_index < 2:
        return None

    version_index = py_tag_index - 1
    distribution_parts = parts[:version_index]
    version = parts[version_index]

    if not distribution_parts or not version:
        return None

    distribution = "-".join(distribution_parts)

    return WheelInfo(
        path=Path(filename),
        file_name=filename,
        normalized_name=normalize_package_name(distribution),
        version=version,
    )


def scan_wheels(root_dir: Path) -> Dict[str, List[WheelInfo]]:
    wheel_index: Dict[str, List[WheelInfo]] = {}

    for wheel_path in root_dir.rglob("*.whl"):
        parsed = parse_wheel_filename(wheel_path.name)
        if not parsed:
            continue

        parsed.path = wheel_path
        wheel_index.setdefault(parsed.normalized_name, []).append(parsed)

    return wheel_index


def choose_best_recursive_match(
    requirement: RequirementEntry,
    candidates: List[WheelInfo],
) -> Tuple[Optional[WheelInfo], bool, str]:
    if not candidates:
        return None, False, "No candidate wheels found during recursive scan."

    if requirement.version:
        exact = [c for c in candidates if c.version == requirement.version]
        if exact:
            exact.sort(key=lambda c: (len(str(c.path)), c.file_name.lower()))
            return exact[0], True, "Exact version match found during recursive scan."

        candidates_sorted = sorted(candidates, key=lambda c: (len(str(c.path)), c.file_name.lower()))
        return candidates_sorted[0], False, "Package found during recursive scan, but version did not match exactly."

    candidates_sorted = sorted(candidates, key=lambda c: (len(str(c.path)), c.file_name.lower()))
    return candidates_sorted[0], False, "Package found during recursive scan; no version specified."


def build_local_requirement_line(requirement: RequirementEntry, wheel_path: Path) -> str:
    return f"{requirement.package_name} @ file:///{wheel_path.as_posix()}"


def resolve_requirement(
    requirement: RequirementEntry,
    wheel_index: Dict[str, List[WheelInfo]],
) -> MatchResult:
    candidates = wheel_index.get(requirement.normalized_name, [])

    # ---------------------------------------------------------
    # 1. Exact file URL path check first
    # ---------------------------------------------------------
    if requirement.source_file_path:
        exact_path = Path(requirement.source_file_path)
        if exact_path.exists() and exact_path.is_file():
            parsed = parse_wheel_filename(exact_path.name)

            if parsed:
                name_matches = parsed.normalized_name == requirement.normalized_name
                version_matches = (requirement.version is None) or (parsed.version == requirement.version)

                if name_matches:
                    return MatchResult(
                        requirement=requirement,
                        found=True,
                        match_strategy="exact_url_path",
                        matched_wheel=str(exact_path),
                        matched_version=parsed.version,
                        exact_version_match=version_matches,
                        candidate_count=len(candidates),
                        candidates=[str(c.path) for c in sorted(candidates, key=lambda x: str(x.path).lower())],
                        note=(
                            "Exact file URL path exists and was used."
                            if version_matches
                            else "Exact file URL path exists and was used, but version did not match extracted requirement version."
                        ),
                    )

                # File exists, but its parsed package name doesn't match requested package name.
                # Fall back to recursive search.
        # If exact path missing, fall through to recursive search.

    # ---------------------------------------------------------
    # 2. Recursive scan fallback
    # ---------------------------------------------------------
    best, exact, note = choose_best_recursive_match(requirement, candidates)

    if best:
        strategy = "recursive_exact_version" if exact else "recursive_name_only"
        return MatchResult(
            requirement=requirement,
            found=True,
            match_strategy=strategy,
            matched_wheel=str(best.path),
            matched_version=best.version,
            exact_version_match=exact,
            candidate_count=len(candidates),
            candidates=[str(c.path) for c in sorted(candidates, key=lambda x: str(x.path).lower())],
            note=note,
        )

    return MatchResult(
        requirement=requirement,
        found=False,
        match_strategy="missing",
        matched_wheel=None,
        matched_version=None,
        exact_version_match=False,
        candidate_count=0,
        candidates=[],
        note="No exact file path match and no recursive wheel match found.",
    )


def analyze_requirements(
    requirements: List[RequirementEntry],
    wheel_index: Dict[str, List[WheelInfo]],
) -> List[MatchResult]:
    return [resolve_requirement(req, wheel_index) for req in requirements]


def parse_requirements_block(text: str) -> List[RequirementEntry]:
    entries: List[RequirementEntry] = []
    for line in text.splitlines():
        parsed = parse_requirement_line(line)
        if parsed:
            entries.append(parsed)
    return entries


def write_report(results: List[MatchResult], report_path: Path) -> None:
    report = {
        "summary": {
            "total": len(results),
            "found": sum(1 for r in results if r.found),
            "missing": sum(1 for r in results if not r.found),
            "exact_version_matches": sum(1 for r in results if r.exact_version_match),
            "exact_url_path_matches": sum(1 for r in results if r.match_strategy == "exact_url_path"),
            "recursive_exact_version_matches": sum(1 for r in results if r.match_strategy == "recursive_exact_version"),
            "recursive_name_only_matches": sum(1 for r in results if r.match_strategy == "recursive_name_only"),
        },
        "results": [asdict(r) for r in results],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_resolved_requirements(results: List[MatchResult], output_path: Path) -> None:
    lines: List[str] = []

    for result in results:
        req = result.requirement
        if result.found and result.matched_wheel:
            lines.append(build_local_requirement_line(req, Path(result.matched_wheel)))
        else:
            lines.append(f"# MISSING WHEEL: {req.raw_line}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(results: List[MatchResult]) -> None:
    total = len(results)
    found = sum(1 for r in results if r.found)
    missing = sum(1 for r in results if not r.found)
    exact_version = sum(1 for r in results if r.exact_version_match)
    exact_url = sum(1 for r in results if r.match_strategy == "exact_url_path")
    recursive_exact = sum(1 for r in results if r.match_strategy == "recursive_exact_version")
    recursive_name_only = sum(1 for r in results if r.match_strategy == "recursive_name_only")

    print("=" * 80)
    print("Wheel scan summary")
    print("=" * 80)
    print(f"Total requirements              : {total}")
    print(f"Found wheels                    : {found}")
    print(f"Missing wheels                  : {missing}")
    print(f"Exact version matches           : {exact_version}")
    print(f"Exact file URL path matches     : {exact_url}")
    print(f"Recursive exact version matches : {recursive_exact}")
    print(f"Recursive name-only matches     : {recursive_name_only}")
    print()

    if missing:
        print("Missing packages:")
        for r in results:
            if not r.found:
                print(f"  - {r.requirement.raw_line}")
        print()

    mismatches = [r for r in results if r.found and not r.exact_version_match]
    if mismatches:
        print("Found packages with non-exact version matches:")
        for r in mismatches:
            print(
                f"  - {r.requirement.package_name} "
                f"(wanted={r.requirement.version}, found={r.matched_version}) "
                f"[{r.match_strategy}] -> {r.matched_wheel}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve requirement entries against wheel files using exact file URL path first, then recursive search."
    )
    parser.add_argument(
        "--wheels-root",
        default=r"E:\wheels",
        help="Root folder to scan recursively for wheel files.",
    )
    parser.add_argument(
        "--report",
        default=r"E:\emtac\services\wheel_scan_report.json",
        help="Path to write the JSON report.",
    )
    parser.add_argument(
        "--resolved-requirements",
        default=r"E:\emtac\services\resolved_local_requirements.txt",
        help="Path to write the resolved local requirements file.",
    )

    args = parser.parse_args()

    wheels_root = Path(args.wheels_root)
    report_path = Path(args.report)
    resolved_requirements_path = Path(args.resolved_requirements)

    if not wheels_root.exists():
        raise FileNotFoundError(f"Wheels root does not exist: {wheels_root}")

    requirements = parse_requirements_block(RAW_REQUIREMENTS)

    print(f"Scanning wheels recursively under: {wheels_root}")
    wheel_index = scan_wheels(wheels_root)

    results = analyze_requirements(requirements, wheel_index)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_requirements_path.parent.mkdir(parents=True, exist_ok=True)

    write_report(results, report_path)
    write_resolved_requirements(results, resolved_requirements_path)
    print_summary(results)

    print(f"JSON report written to: {report_path}")
    print(f"Resolved requirements written to: {resolved_requirements_path}")


if __name__ == "__main__":
    main()