from __future__ import annotations

import re
import time
from typing import List, Optional

from PIL import Image
import torch

from app.config.gpu_logger import gpu_phase, gpu_debug, gpu_warning


class VLMService:

    SYSTEM_RULES: str = (
        "You are an OCR/transcription engine.\n"
        "Return ONLY the page content as Markdown.\n"
        "No analysis, no reasoning, no <think>.\n"
        "Do not describe the page.\n"
        "If the page is blank, return an empty string.\n"
    )

    _THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
    _THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
    _ROLE_LINE_RE = re.compile(
        r"^(system|user|assistant)\s*:?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    _LEADING_ASSISTANT_RE = re.compile(
        r"^\s*assistant\s*\n+",
        re.IGNORECASE,
    )

    # ------------------------------------------------------------
    # Single image inference
    # ------------------------------------------------------------

    def image_to_text(
        self,
        *,
        model,
        processor,
        image_path: str,
        prompt: str,
        max_new_tokens: int,
        request_id: str,
    ) -> str:

        rid = request_id
        t0 = time.time()

        gpu_debug(f"VLM single start | image={image_path}", rid)

        with Image.open(image_path) as img:
            img.load()
            image = img.convert("RGB")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.SYSTEM_RULES}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
        ]

        chat_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=[chat_text],
            images=[image],
            return_tensors="pt",
        )

        tokenizer = processor.tokenizer

        t_infer = time.time()

        try:

            with torch.no_grad():

                output = model.generate(
                    **inputs,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

        finally:

            gpu_phase("VLM_GENERATE", time.time() - t_infer, rid)

        # ----------------------------
        # FIX: remove prompt tokens
        # ----------------------------

        input_length = inputs["input_ids"].shape[1]

        decoded = processor.batch_decode(
            output[:, input_length:],  # only generated tokens
            skip_special_tokens=True,
        )[0]

        gpu_debug(f"VLM raw output | page={image_path} chars={len(decoded)}", rid)
        gpu_debug(decoded[:2000], rid)

        cleaned = self._clean_vlm_output(decoded, prompt=prompt)

        gpu_phase("VLM_IMAGE_TO_TEXT", time.time() - t0, rid)

        gpu_debug(f"VLM single complete | chars={len(cleaned)}", rid)

        return cleaned

    # ------------------------------------------------------------
    # Batched inference
    # ------------------------------------------------------------

    def images_to_text(
        self,
        *,
        model,
        processor,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int,
        request_id: str,
        max_batch_size: Optional[int] = None,
    ) -> List[str]:

        rid = request_id
        t0 = time.time()

        n = len(image_paths)

        if n == 0:
            return []

        if max_batch_size and max_batch_size > 0:
            bs = int(max_batch_size)
        else:
            bs = min(4, n)

        bs = min(bs, n)

        gpu_debug(f"VLM batch start | total={n} initial_bs={bs}", rid)

        results: List[str] = [""] * n

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.SYSTEM_RULES}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
        ]

        chat_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        tokenizer = processor.tokenizer

        def _clear_cuda():

            if torch.cuda.is_available():

                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

        i = 0

        while i < n:

            cur_bs = min(bs, n - i)

            while True:

                chunk_paths = image_paths[i: i + cur_bs]

                images: List[Image.Image] = []

                try:

                    for p in chunk_paths:

                        with Image.open(p) as img:
                            img.load()
                            images.append(img.convert("RGB"))

                    texts = [chat_text] * len(images)

                    inputs = processor(
                        text=texts,
                        images=images,
                        return_tensors="pt",
                        padding=True,
                    )

                    t_infer = time.time()

                    try:

                        with torch.no_grad():

                            output = model.generate(
                                **inputs,
                                max_new_tokens=int(max_new_tokens),
                                do_sample=False,
                                use_cache=True,
                                pad_token_id=tokenizer.eos_token_id,
                            )

                    finally:

                        gpu_phase("VLM_GENERATE", time.time() - t_infer, rid)

                    # ----------------------------
                    # FIX: remove prompt tokens
                    # ----------------------------

                    input_length = inputs["input_ids"].shape[1]

                    decoded_list = processor.batch_decode(
                        output[:, input_length:],
                        skip_special_tokens=True,
                    )

                    for idx, raw in enumerate(decoded_list):
                        gpu_debug(
                            f"VLM raw batch output | start={i} idx={idx} chars={len(raw)}",
                            rid,
                        )
                        gpu_debug(raw[:2000], rid)

                    cleaned_list = [
                        self._clean_vlm_output(d, prompt=prompt)
                        for d in decoded_list
                    ]

                    for j, txt in enumerate(cleaned_list):
                        results[i + j] = txt

                    gpu_debug(f"VLM batch chunk ok | start={i} bs={cur_bs}", rid)

                    i += cur_bs

                    break

                except torch.cuda.OutOfMemoryError:

                    gpu_warning(
                        f"CUDA OOM in batch chunk | start={i} bs={cur_bs} -> reducing batch",
                        rid,
                    )

                    _clear_cuda()

                    if cur_bs <= 1:
                        raise

                    cur_bs = max(1, cur_bs // 2)

                finally:

                    images.clear()

                    try:
                        del inputs
                    except Exception:
                        pass

                    try:
                        del output
                    except Exception:
                        pass

        gpu_phase("VLM_IMAGE_TO_TEXT", time.time() - t0, rid)

        gpu_debug(f"VLM batch complete | total={n} initial_bs={bs}", rid)

        return results

    # ------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------

    def _clean_vlm_output(self, text: str, *, prompt: str) -> str:

        if not text:
            return ""

        t = text.strip()

        # ------------------------------------------------------------
        # Remove reasoning blocks
        # ------------------------------------------------------------
        t = self._THINK_BLOCK_RE.sub("", t).strip()
        t = self._THINK_OPEN_RE.sub("", t).strip()

        # ------------------------------------------------------------
        # Remove role markers
        # ------------------------------------------------------------
        t = self._LEADING_ASSISTANT_RE.sub("", t).strip()
        t = self._ROLE_LINE_RE.sub("", t).strip()

        # ------------------------------------------------------------
        # Remove echoed system prompt
        # ------------------------------------------------------------
        sys_rules = self.SYSTEM_RULES.strip()

        if sys_rules and t.startswith(sys_rules):
            t = t[len(sys_rules):].lstrip()

        # ------------------------------------------------------------
        # Remove echoed user prompt
        # ------------------------------------------------------------
        p = (prompt or "").strip()

        if p and t.startswith(p):
            t = t[len(p):].lstrip()

        # ------------------------------------------------------------
        # Remove markdown fences (common VLM artifact)
        # ------------------------------------------------------------

        # opening fence
        if t.startswith("```markdown"):
            t = t[len("```markdown"):].lstrip()

        if t.startswith("```"):
            t = t[len("```"):].lstrip()

        # closing fence
        if t.endswith("```"):
            t = t[:-3].rstrip()

        # lone fence case (blank page artifact)
        if t.strip() in {"```", "````"}:
            return ""

        # ------------------------------------------------------------
        # Normalize whitespace
        # ------------------------------------------------------------
        t = re.sub(r"\n{3,}", "\n\n", t).strip()

        return t


VLM_SERVICE = VLMService()