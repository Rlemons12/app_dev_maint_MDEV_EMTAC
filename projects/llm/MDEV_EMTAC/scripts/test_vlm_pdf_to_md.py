import argparse
from pathlib import Path
import torch

from modules.vlm.vlm_runtime_adapter import VLMRuntimeAdapter


MODEL_PATH = r"E:\emtac\models\llm\nu_markdown\models\NuMarkdown-8B-Thinking"


SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
}


def print_gpu_info():
    print("--------------------------------------------------")
    print("GPU Info")
    print("--------------------------------------------------")
    print("CUDA Available:", torch.cuda.is_available())
    print("GPU Count     :", torch.cuda.device_count())

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} Name   :", torch.cuda.get_device_name(i))

    print("--------------------------------------------------")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, help="PDF or image file")
    parser.add_argument("--out", required=True)

    parser.add_argument("--pages", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=384)
    parser.add_argument("--debug-vram", action="store_true")

    args = parser.parse_args()

    input_path = Path(args.input)
    out_path = Path(args.out)

    if not input_path.exists():
        raise RuntimeError(f"Input file does not exist: {input_path}")

    if args.debug_vram:
        print_gpu_info()

    print("--------------------------------------------------")
    print("DIRECT VLM Runtime Test")
    print("--------------------------------------------------")
    print(f"Model Path  : {MODEL_PATH}")
    print(f"Input File  : {input_path}")
    print(f"MD Output   : {out_path}")
    print(f"Max Pages   : {args.pages}")
    print(f"DPI         : {args.dpi}")
    print(f"Max Side    : {args.max_side}")
    print(f"Max Tokens  : {args.tokens}")
    print("--------------------------------------------------")

    adapter = VLMRuntimeAdapter(
        model_path=MODEL_PATH,
        max_new_tokens=args.tokens,
    )

    suffix = input_path.suffix.lower()

    # --------------------------------------------------
    # Auto-detect input type
    # --------------------------------------------------
    if suffix == ".pdf":
        markdown = adapter.pdf_to_markdown(
            pdf_path=str(input_path),
            max_pages=args.pages,
            dpi=args.dpi,
            max_side=args.max_side,
        )

    elif suffix in SUPPORTED_IMAGE_EXTENSIONS:
        markdown = adapter.image_to_markdown(
            image_path=str(input_path),
            max_side=args.max_side,
        )

    else:
        raise RuntimeError(
            f"Unsupported file type: {suffix}. "
            f"Supported: PDF or {SUPPORTED_IMAGE_EXTENSIONS}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")

    print(f"✔ Saved to: {out_path}")


if __name__ == "__main__":
    main()
