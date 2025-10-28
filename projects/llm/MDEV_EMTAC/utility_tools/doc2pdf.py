from docx2pdf import convert

# Path to your Word document
word_document_path = r"C:\Users\10169062\Desktop\AI_EMTACr7c6\Docs\TidlandRebuildInstructions.docx"

# Path to save the PDF
pdf_output_path = r"C:\Users\10169062\Desktop\AI_EMTACr7c6\Docs\document.pdf"

# Convert the Word document to PDF
convert(word_document_path, pdf_output_path)

print("Conversion complete!")
