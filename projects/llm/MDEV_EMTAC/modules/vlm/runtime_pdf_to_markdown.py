"""
runtime_pdf_to_markdown.py

Minimal PDF -> Markdown converter using a Vision-Language Model.

No:
- Stage schemas
- Database
- Training pipeline
- Env loading
- ModelsConfig
- Structured JSON

Just:
PDF -> Markdown string
"""

from pathlib import Path
from typing import List
import tempfile
import os

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

import fitz  # PyMuPDF


# ============================================================
# Runtime VLM PDF Converter
# ============================================================

class RuntimePDFToMarkdown:

    def __init__(
        self,
        model_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        max_new_tokens: int = 1800,
        max_image_long_side: int = 2048,
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.max_image_long_side = max_image_long_side

        # Force offline if you're air-gapped
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        self.model.eval()

    # --------------------------------------------------------
    # PDF Rendering (PyMuPDF only)
    # --------------------------------------------------------

    def _render_pdf_to_images(self, pdf_path: str) -> List[str]:

        pdf_path = str(pdf_path)
        doc = fitz.open(pdf_path)

        temp_dir = Path(tempfile.mkdtemp())
        image_paths = []

        for i in range(doc.page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)

            img_path = temp_dir / f"page_{i+1:04d}.png"
            pix.save(str(img_path))
            image_paths.append(str(img_path))

        doc.close()
        return image_paths

    # --------------------------------------------------------
    # Single Page -> Markdown
    # --------------------------------------------------------

    def _image_to_markdown(self, image_path: str) -> str:

        img = Image.open(image_path).convert("RGB")

        # Resize guard
        w, h = img.size
        long_side = max(w, h)
        if long_side > self.max_image_long_side:
            scale = self.max_image_long_side / long_side
            img = img.resize((int(w * scale), int(h * scale)))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            "Convert this scanned document page into clean structured Markdown.\n"
                            "Preserve headings, tables, lists, indentation exactly.\n"
                            "Do NOT include reasoning.\n"
                            "Output ONLY the Markdown."
                        ),
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[img],
            return_tensors="pt",
        )

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
            )

        generated_ids = output_ids[:, inputs["input_ids"].size(1):]

        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()

        if "<think>" in output_text:
            output_text = output_text.split("</think>")[-1].strip()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_text

    # --------------------------------------------------------
    # Public Method
    # --------------------------------------------------------

    def convert_pdf(self, pdf_path: str) -> str:

        if not Path(pdf_path).exists():
            raise FileNotFoundError(pdf_path)

        image_paths = self._render_pdf_to_images(pdf_path)

        markdown_pages = []

        for img_path in image_paths:
            md = self._image_to_markdown(img_path)
            markdown_pages.append(md)

        return "\n\n".join(markdown_pages)
