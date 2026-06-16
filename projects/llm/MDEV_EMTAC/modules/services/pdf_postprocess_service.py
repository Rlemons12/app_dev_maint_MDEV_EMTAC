import os
from typing import Callable, Optional


class PdfPostprocessService:
    def __init__(
        self,
        dark_threshold: float = 0.45,
        min_luminance: float = 0.72,
    ) -> None:
        self.dark_threshold = dark_threshold
        self.min_luminance = min_luminance

    @staticmethod
    def _log(logger: Optional[Callable[[str], None]], message: str) -> None:
        if logger:
            logger(message)

    @staticmethod
    def _luminance(r: float, g: float, b: float) -> float:
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _lift_rgb_on_black(self, r: float, g: float, b: float) -> tuple[float, float, float]:
        r = self._clamp01(r)
        g = self._clamp01(g)
        b = self._clamp01(b)
        lum = self._luminance(r, g, b)
        if lum >= self.dark_threshold:
            return r, g, b

        if lum <= 1e-6:
            return self.min_luminance, self.min_luminance, self.min_luminance

        scale = self.min_luminance / lum
        return self._clamp01(r * scale), self._clamp01(g * scale), self._clamp01(b * scale)

    def apply_black_background(
        self,
        pdf_path: str,
        logger: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """
        Inject a black rectangle as the first draw command on every PDF page and
        brighten very dark vector colors so geometry stays visible on black.
        """
        try:
            from pypdf import PdfReader, PdfWriter
            from pypdf.generic import ContentStream, DecodedStreamObject, FloatObject
        except Exception:
            self._log(
                logger,
                "Black background post-process requires 'pypdf'. Install with: py -3.10 -m pip install pypdf",
            )
            return False

    def should_apply_black_background_auto(
        self,
        pdf_path: str,
        logger: Optional[Callable[[str], None]] = None,
        max_pages: int = 2,
    ) -> bool:
        """
        Decide whether a white-background PDF likely has low-contrast bright CAD geometry
        and should be switched to black background.
        """
        try:
            import fitz
        except Exception:
            self._log(
                logger,
                "Auto background mode requires 'PyMuPDF'. Install with: py -3.10 -m pip install PyMuPDF",
            )
            return False

        pdf_path = os.path.abspath(os.path.expanduser(pdf_path))
        if not os.path.exists(pdf_path):
            self._log(logger, f"Cannot analyze PDF; file not found: {pdf_path}")
            return False

        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            self._log(logger, f"Cannot open PDF for auto background analysis: {exc}")
            return False

        try:
            total_px = 0
            white_px = 0
            content_px = 0
            bright_content_px = 0
            dark_content_px = 0
            neon_content_px = 0

            pages_to_check = min(max_pages, len(doc))
            for page_index in range(pages_to_check):
                page = doc[page_index]
                # Low-res render is sufficient for contrast heuristics.
                pix = page.get_pixmap(matrix=fitz.Matrix(0.35, 0.35), alpha=False)
                data = pix.samples
                step_px = 4  # sample every 4th pixel to keep analysis cheap
                step = 3 * step_px

                for i in range(0, len(data), step):
                    r = data[i]
                    g = data[i + 1]
                    b = data[i + 2]
                    total_px += 1

                    if r >= 245 and g >= 245 and b >= 245:
                        white_px += 1
                        continue

                    content_px += 1
                    lum = self._luminance(r / 255.0, g / 255.0, b / 255.0)
                    if lum >= 0.68:
                        bright_content_px += 1
                    if lum <= 0.35:
                        dark_content_px += 1

                    is_green_cyan = g >= 150 and r <= 170 and b <= 170
                    is_yellow = r >= 170 and g >= 170 and b <= 140
                    neon_content_px += 1 if (is_green_cyan or is_yellow) else 0

            if total_px == 0 or content_px == 0:
                return False

            white_ratio = white_px / total_px
            bright_ratio = bright_content_px / content_px
            dark_ratio = dark_content_px / content_px
            neon_ratio = neon_content_px / content_px

            decision = bool(
                white_ratio >= 0.55
                and bright_ratio >= 0.55
                and dark_ratio <= 0.18
                and neon_ratio >= 0.08
            )
            self._log(
                logger,
                (
                    f"Auto background analysis for {os.path.basename(pdf_path)}: "
                    f"white={white_ratio:.2f}, bright={bright_ratio:.2f}, "
                    f"dark={dark_ratio:.2f}, neon={neon_ratio:.2f}, decision={'black' if decision else 'white'}"
                ),
            )
            return decision
        finally:
            doc.close()

    def auto_apply_black_background_if_needed(
        self,
        pdf_path: str,
        logger: Optional[Callable[[str], None]] = None,
    ) -> tuple[bool, bool]:
        """
        Returns (applied, decision_black).
        """
        decision_black = self.should_apply_black_background_auto(pdf_path, logger=logger)
        if not decision_black:
            return False, False
        applied = self.apply_black_background(pdf_path, logger=logger)
        return applied, True

        pdf_path = os.path.abspath(os.path.expanduser(pdf_path))
        if not os.path.exists(pdf_path):
            self._log(logger, f"Cannot apply black background; PDF not found: {pdf_path}")
            return False

        temp_path = pdf_path + ".bg.tmp"

        try:
            reader = PdfReader(pdf_path)
            writer = PdfWriter()

            for page in reader.pages:
                writer.add_page(page)
                out_page = writer.pages[-1]
                width = float(out_page.mediabox.width)
                height = float(out_page.mediabox.height)

                modified_bytes = None
                try:
                    contents = out_page.get_contents()
                    if contents is not None:
                        content_stream = ContentStream(contents, writer)
                        updated_operations = []
                        for operands, operator in content_stream.operations:
                            if operator in (b"rg", b"RG") and len(operands) >= 3:
                                r, g, b = (float(operands[0]), float(operands[1]), float(operands[2]))
                                nr, ng, nb = self._lift_rgb_on_black(r, g, b)
                                operands = [FloatObject(nr), FloatObject(ng), FloatObject(nb)]
                            elif operator in (b"g", b"G") and len(operands) >= 1:
                                gray = float(operands[0])
                                ng = self._lift_rgb_on_black(gray, gray, gray)[0]
                                operands = [FloatObject(ng)]
                            updated_operations.append((operands, operator))
                        content_stream.operations = updated_operations
                        modified_bytes = content_stream.get_data()
                except Exception as exc:
                    self._log(logger, f"Color lift step skipped for one page: {exc}")

                if modified_bytes is None:
                    try:
                        modified_bytes = out_page._get_contents_as_bytes()
                    except Exception:
                        modified_bytes = b""

                bg_bytes = f"q\n0 0 0 rg\n0 0 {width:.4f} {height:.4f} re f\nQ\n".encode("ascii")
                combined = DecodedStreamObject()
                combined.set_data(bg_bytes + modified_bytes)
                out_page.replace_contents(combined)

            with open(temp_path, "wb") as handle:
                writer.write(handle)

            os.replace(temp_path, pdf_path)
            return True
        except Exception as exc:
            self._log(logger, f"Failed black-background post-process for {pdf_path}: {exc}")
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
            return False
