# modules/report.py
import os
from fpdf import FPDF
from datetime import datetime

REPORT_DIR = "outputs/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

class PDFReport:
    def __init__(self, title="InsightX AI Report"):
        self.pdf = FPDF()
        self.title = title

    def add_page(self):
        self.pdf.add_page()
        self.pdf.set_font("Arial", 'B', 16)
        self.pdf.cell(0, 10, self.title, ln=True, align='C')
        self.pdf.ln(6)

    def add_paragraph(self, text, size=11):
        self.pdf.set_font("Arial", size=size)
        self.pdf.set_left_margin(10)
        self.pdf.set_right_margin(10)
        self.pdf.multi_cell(0, 6, text)
        self.pdf.ln(2)

    def add_image(self, image_path, w=170):
        if image_path and os.path.exists(image_path):
            try:
                self.pdf.image(image_path, w=w)
                self.pdf.ln(4)
            except Exception as e:
                # if image is html or not supported, skip
                pass

    def save(self, filename=None):
        if filename is None:
            filename = f"{REPORT_DIR}/InsightX_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        self.pdf.output(filename)
        return filename

    def build(self, overview_text, insights_list, image_paths):
        self.add_page()
        self.add_paragraph("OVERVIEW:")
        self.add_paragraph(overview_text)
        self.add_paragraph("KEY INSIGHTS:")
        for ins in insights_list:
            self.add_paragraph("- " + ins)
        self.add_paragraph("VISUALIZATIONS:")
        for img in image_paths:
            # if html plot saved, skip image insertion (FPDF needs PNG/JPG)
            if img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg"):
                self.add_image(img)
        return self.save()