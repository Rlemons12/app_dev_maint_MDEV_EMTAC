import numpy as np
import os
from pdf2image import convert_from_path
import easyocr
from tkinter import Tk, filedialog

# Set your Poppler binary path
POPLER_PATH = r"C:\Users\10169062\Desktop\AU_IndusMaintdb\Database\Release-24.08.0-0\poppler-24.08.0\Library\bin"


def prompt_for_pdf():
    Tk().withdraw()  # Hide root window
    file_path = filedialog.askopenfilename(
        title="Select a PDF file",
        filetypes=[("PDF files", "*.pdf")]
    )
    return file_path

def pdf_to_images(pdf_path):
    print("Converting PDF to images...")
    return convert_from_path(pdf_path, dpi=300, poppler_path=POPLER_PATH)

def ocr_images_with_easyocr(images):
    import numpy as np
    reader = easyocr.Reader(['en'])
    extracted_text = ""

    for i, image in enumerate(images):
        print(f"OCR on page {i+1}...")

        # ✅ Convert PIL image to NumPy array
        image_np = np.array(image)

        results = reader.readtext(image_np, detail=0)
        page_text = '\n'.join(results)
        extracted_text += f"\n\n=== Page {i+1} ===\n\n{page_text}"

    return extracted_text

def save_text(output_text, pdf_path):
    base = os.path.splitext(pdf_path)[0]
    output_path = base + "_OCR.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output_text)
    print(f"✅ OCR text saved to: {output_path}")

def main():
    pdf_path = prompt_for_pdf()
    if not pdf_path:
        print("No PDF selected.")
        return

    images = pdf_to_images(pdf_path)
    text = ocr_images_with_easyocr(images)
    save_text(text, pdf_path)

if __name__ == "__main__":
    main()
