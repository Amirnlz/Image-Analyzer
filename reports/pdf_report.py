from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


class PDFReport:
    def __init__(self, color_ranges, original_image_path, modified_image_path):
        self.color_ranges = color_ranges
        self.original_image_path = original_image_path
        self.modified_image_path = modified_image_path

    def generate(self, filename="color_report.pdf"):
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter
        y = height - 50

        # Insert Original Image
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Original Image:")
        y -= 20
        orig_img = ImageReader(self.original_image_path)
        c.drawImage(orig_img, 50, y - 200, width=200, height=200)
        y -= 220

        # Insert Modified Image
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Modified Image:")
        y -= 20
        mod_img = ImageReader(self.modified_image_path)
        c.drawImage(mod_img, 50, y - 200, width=200, height=200)
        y -= 220

        c.setFont("Helvetica", 12)
        c.drawString(50, y, "Color Report")
        y -= 20

        for idx, color_range in enumerate(self.color_ranges):
            c.drawString(50, y, f"Color Range {idx + 1}:")
            y -= 15
            for color in color_range:
                hex_color = '#%02x%02x%02x' % tuple(color)
                r, g, b = [value / 255 for value in color]
                c.setFillColorRGB(r, g, b)
                c.rect(60, y, 50, 10, fill=1)
                c.setFillColorRGB(0, 0, 0)
                c.drawString(115, y + 2, hex_color)
                y -= 15
                if y < 50:
                    c.showPage()
                    y = height - 50
        c.save()
