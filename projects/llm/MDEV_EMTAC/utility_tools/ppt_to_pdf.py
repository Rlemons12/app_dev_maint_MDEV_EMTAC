import os
import comtypes.client

# Function to convert PPTX to PDF
def convert_pptx_to_pdf(pptx_file, pdf_file):
    powerpoint = comtypes.client.CreateObject("PowerPoint.Application")
    presentation = powerpoint.Presentations.Open(pptx_file)
    presentation.SaveAs(pdf_file, 32)  # 32 represents PDF format
    presentation.Close()
    powerpoint.Quit()

# Specify the PPTX file and output PDF file
pptx_file = r'C:\Users\10169062\Desktop\AI_EMTACr3r4\PPT_FILES\debug_hands_on2.pptx'
pdf_file = r'C:\Users\10169062\Desktop\AI_EMTACr3r4\PPT_FILES\output.pdf'

# Call the function to convert PPTX to PDF
convert_pptx_to_pdf(pptx_file, pdf_file)

print("Conversion complete!")
