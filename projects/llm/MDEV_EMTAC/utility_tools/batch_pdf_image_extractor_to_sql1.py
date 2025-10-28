import fitz
import os
from modules.configuration.config import PDF_FOR_EXTRACTION_FOLDER, IMAGES_EXTRACTED


def extract_images_from_pdf(pdf_path, output_folder):
    print(f"Opening PDF file from: {pdf_path}")
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"Total pages in the PDF: {total_pages}")

    extracted_images = []
    sql_statements = []

    for page_num in range(total_pages):
        page = doc[page_num]
        img_list = page.get_images(full=True)

        print(f"Processing page {page_num + 1}/{total_pages} with {len(img_list)} images.")

        for img_index, img in enumerate(img_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            image_filename = f"{os.path.basename(pdf_path).replace('.pdf', '')}_page_{page_num + 1}_img_{img_index + 1}.png"
            image_path = os.path.join(output_folder, image_filename)

            extracted_images.append(image_filename)

            print(f"Saving image to {image_path}")

            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

            sql = f"INSERT INTO pdf_images (pdf_name, page_number, image_filename, image_path) VALUES ('{os.path.basename(pdf_path)}', {page_num + 1}, '{image_filename}', '{image_path}');"
            sql_statements.append(sql)

    return sql_statements

def process_folder(pdf_folder, output_folder):
    # Check and create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all the PDF files in the directory
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

    all_sql_statements = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        sql_statements = extract_images_from_pdf(pdf_path, output_folder)
        all_sql_statements.extend(sql_statements)

    with open(os.path.join(output_folder, "insertion_script.sql"), "w") as f:
        for statement in all_sql_statements:
            f.write(statement + "\n")

    print("PDF image extraction for the entire folder is complete.")

# Using the configuration variables for folder paths
pdf_folder = PDF_FOR_EXTRACTION_FOLDER
image_output_folder = IMAGES_EXTRACTED


process_folder(pdf_folder, image_output_folder)
print("SQL insertion script has been generated.")
