import os
from plyer import filechooser as plyer_filechooser
import platform
from kivy.uix.modalview import ModalView
from kivy.uix.boxlayout import BoxLayout
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDRaisedButton, MDFlatButton
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.fitimage import FitImage
from kivymd.uix.label import MDLabel
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from plyer import filechooser
from PIL import Image as PILImage
from modules.configuration.log_config import logger
from kivy.uix.modalview import ModalView
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import AsyncImage
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel

from kivy.uix.modalview import ModalView
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import AsyncImage
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDRaisedButton
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import os


class PartDetailsPopup(ModalView):
    def __init__(self, part, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (0.95, 0.95)
        self.auto_dismiss = True
        self.part = part

        layout = BoxLayout(orientation='vertical', spacing=15, padding=20)

        # Image display (bigger)
        if hasattr(part, 'image_path') and part.image_path:
            layout.add_widget(AsyncImage(
                source=part.image_path,
                size_hint=(1, 0.6),
                allow_stretch=True
            ))

        # Part info
        layout.add_widget(MDLabel(
            text=f"Part: {getattr(part, 'part_number', 'N/A')}",
            theme_text_color="Custom", text_color=(0, 1, 1, 1),
            halign="center", size_hint=(1, 0.1)
        ))
        layout.add_widget(MDLabel(
            text=f"Description: {getattr(part, 'name', 'N/A')}",
            theme_text_color="Custom", text_color=(0, 1, 1, 1),
            halign="center", size_hint=(1, 0.1)
        ))

        # Buttons
        button_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.2), spacing=20, padding=(20, 10))

        save_btn = MDRaisedButton(text="Save as PDF", on_release=self.save_to_pdf)
        close_btn = MDRaisedButton(text="Close", on_release=self.close_popup)

        button_layout.add_widget(save_btn)
        button_layout.add_widget(close_btn)

        layout.add_widget(button_layout)
        self.add_widget(layout)

    def close_popup(self, *args):
        logger.debug("Close button clicked.")
        self.dismiss()

    def save_to_pdf(self, *args):
        filechooser_path = filechooser.save_file(title="Save PDF As...", filters=[("PDF Files", "*.pdf")])

        if not filechooser_path:
            logger.debug("Save cancelled.")
            return

        pdf_path = filechooser_path[0]
        if not pdf_path.lower().endswith(".pdf"):
            pdf_path += ".pdf"

        c = canvas.Canvas(pdf_path, pagesize=A4)
        width, height = A4

        y = height - 50
        spacing = 20

        c.drawString(100, y, f"Part Number: {getattr(self.part, 'part_number', 'N/A')}")
        y -= spacing
        c.drawString(100, y, f"Description: {getattr(self.part, 'name', 'N/A')}")
        y -= spacing
        c.drawString(100, y, f"OEM/MFG: {getattr(self.part, 'oem_mfg', 'N/A')}")
        y -= spacing
        c.drawString(100, y, f"Model: {getattr(self.part, 'model', 'N/A')}")
        y -= spacing
        c.drawString(100, y, f"Category: {getattr(self.part, 'class_flag', 'N/A')}")
        y -= spacing
        c.drawString(100, y, f"UD6: {getattr(self.part, 'ud6', 'N/A')}")
        y -= spacing
        c.drawString(100, y, f"Type: {getattr(self.part, 'type', 'N/A')}")
        y -= spacing
        c.drawString(100, y, f"Documentation: {getattr(self.part, 'documentation', 'N/A')}")

        notes = getattr(self.part, 'notes', '')
        if notes:
            y -= spacing
            c.drawString(100, y, "Notes:")
            textobject = c.beginText(100, y - 20)
            for line in notes.split('\n'):
                textobject.textLine(line)
            c.drawText(textobject)
            y = textobject.getY()

        # Add image (larger, centered)
        if getattr(self.part, 'image_path', None) and os.path.exists(self.part.image_path):
            try:
                img_width = 400
                img_height = 400
                img_x = (width - img_width) / 2
                img_y = 100
                c.drawImage(self.part.image_path, img_x, img_y, width=img_width, height=img_height, preserveAspectRatio=True)
            except Exception as e:
                c.drawString(100, 100, f"Image load failed: {str(e)}")
        else:
            c.drawString(100, 100, "No image found")

        c.save()
        logger.debug(f"Saved PDF to: {pdf_path}")

    def on_close(self, *args):
        logger.debug("Popup dismissed.")

# === PDF Export Functions ===
def save_to_pdf(self, *args):
    filechooser_path = filechooser.save_file(title="Save PDF As...", filters=[("PDF Files", "*.pdf")])

    if not filechooser_path:
        logger.debug("Save cancelled.")
        return

    pdf_path = filechooser_path[0]
    if not pdf_path.lower().endswith(".pdf"):
        pdf_path += ".pdf"

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    # Text fields
    y = height - 50
    line_spacing = 20
    c.drawString(100, y, f"Part Number: {getattr(self.part, 'part_number', 'N/A')}")
    y -= line_spacing
    c.drawString(100, y, f"Name (Description): {getattr(self.part, 'name', 'N/A')}")
    y -= line_spacing
    c.drawString(100, y, f"OEM/MFG: {getattr(self.part, 'oem_mfg', 'N/A')}")
    y -= line_spacing
    c.drawString(100, y, f"Model (MFG Part #): {getattr(self.part, 'model', 'N/A')}")
    y -= line_spacing
    c.drawString(100, y, f"Category: {getattr(self.part, 'class_flag', 'N/A')}")
    y -= line_spacing
    c.drawString(100, y, f"UD6: {getattr(self.part, 'ud6', 'N/A')}")
    y -= line_spacing
    c.drawString(100, y, f"Type: {getattr(self.part, 'type', 'N/A')}")
    y -= line_spacing
    c.drawString(100, y, f"Documentation: {getattr(self.part, 'documentation', 'N/A')}")

    notes = getattr(self.part, 'notes', '')
    if notes:
        y -= line_spacing
        c.drawString(100, y, "Notes:")
        textobject = c.beginText(100, y - 20)
        for line in notes.split('\n'):
            textobject.textLine(line)
        c.drawText(textobject)
        y = textobject.getY()

    # Add larger image (if present) centered on page
    if getattr(self.part, 'image_path', None) and os.path.exists(self.part.image_path):
        try:
            img_width = 400
            img_height = 400
            img_x = (width - img_width) / 2
            img_y = 100
            c.drawImage(self.part.image_path, img_x, img_y, width=img_width, height=img_height,
                        preserveAspectRatio=True)
        except Exception as e:
            c.drawString(100, 100, f"Image load failed: {str(e)}")
    else:
        c.drawString(100, 100, "No image found")

    c.save()
    logger.debug(f"Saved PDF to: {pdf_path}")

def show_export_options(widget, part_name, image_path, part_info):
    def handle_export(mode):
        dialog.dismiss()
        save_widget_as_pdf(widget, part_name=part_name, export_mode=mode, image_path=image_path, part_info=part_info)

    dialog = MDDialog(
        title="Export Options",
        text="Choose what to include in the PDF:",
        buttons=[
            MDFlatButton(text="Full Details", on_release=lambda x: handle_export("full")),
            MDFlatButton(text="Image Only", on_release=lambda x: handle_export("image")),
            MDFlatButton(text="Info Only", on_release=lambda x: handle_export("info")),
            MDRaisedButton(text="Cancel", on_release=lambda x: dialog.dismiss())
        ]
    )
    dialog.open()

# === Utility to open popup ===
def open_part_details_popup(part):
    image_path = getattr(part, "image_path", None)
    part_number = getattr(part, "part_number", "N/A")
    description = getattr(part, "description", "N/A")

    part_info = f"Part Number: {part_number}\nDescription: {description}"

    content = MDBoxLayout(orientation="vertical", spacing=10, padding=10)

    if image_path:
        content.add_widget(FitImage(source=image_path, size_hint=(1, 0.6)))

    if part_info:
        content.add_widget(MDLabel(text=part_info, halign="left", size_hint=(1, 0.3)))

    content.add_widget(MDRaisedButton(
        text="Download as PDF",
        on_release=lambda x: show_export_options(content, part_number, image_path, part_info)
    ))

    content.add_widget(MDRaisedButton(
        text="Close",
        on_release=lambda x: popup.dismiss()
    ))

    popup = Popup(
        title=f"Details for {part_number}",
        content=content,
        size_hint=(0.8, 0.8),
        auto_dismiss=True
    )
    popup.open()

class ImageDetailsPopup(ModalView):
    def __init__(self, image_obj, image_path, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (0.9, 0.9)
        self.auto_dismiss = True
        self.image = image_obj
        self.image_path = image_path

        layout = BoxLayout(orientation='vertical', spacing=15, padding=20)

        # Image display (larger view)
        layout.add_widget(AsyncImage(
            source=image_path,
            allow_stretch=True,
            size_hint=(1, 0.7)
        ))

        # Title & Description
        layout.add_widget(MDLabel(
            text=f"{image_obj.title or 'Untitled Image'}",
            halign="center",
            theme_text_color="Custom",
            text_color=(0, 1, 1, 1),
            size_hint=(1, 0.1)
        ))

        layout.add_widget(MDLabel(
            text=image_obj.description or "No description available.",
            halign="center",
            theme_text_color="Custom",
            text_color=(1, 1, 1, 1),
            size_hint=(1, 0.1)
        ))

        # Buttons
        button_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1), spacing=20)

        save_image_btn = MDRaisedButton(text="Save Image As...", on_release=self.save_image)
        close_btn = MDRaisedButton(text="Close", on_release=lambda x: self.dismiss())

        button_layout.add_widget(save_image_btn)
        button_layout.add_widget(close_btn)
        layout.add_widget(button_layout)

        self.add_widget(layout)

    def save_image(self, *args):
        if platform == 'win' or platform == 'linux' or platform == 'macosx':
            # Desktop filechooser
            selected = plyer_filechooser.save_file(title="Save Image As", filters=["*.jpg", "*.jpeg", "*.png", "*.bmp"])
        else:
            # Mobile support
            selected = plyer_filechooser.save_file()

        if selected:
            save_path = selected[0]
            _, ext = os.path.splitext(save_path)
            if not ext:
                # Fallback to original image's extension
                original_ext = os.path.splitext(self.image_path)[1]
                save_path += original_ext

            try:
                shutil.copyfile(self.image_path, save_path)
                print(f"Image saved to: {save_path}")
            except Exception as e:
                print(f"Failed to save image: {e}")