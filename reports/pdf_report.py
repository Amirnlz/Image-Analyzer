from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color
from reportlab.lib.utils import ImageReader


class PDFReport:
    def __init__(self, color_ranges, original_image_path, processed_image_path):
        self.color_ranges = color_ranges
        self.original_image_path = original_image_path
        self.processed_image_path = processed_image_path

    def generate(self, filename="color_report.pdf"):
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter
        y = height - 50
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, "Color Report")
        y -= 30

        # Insert original image
        c.setFont("Helvetica", 12)
        c.drawString(50, y, "Original Image:")
        y -= 15
        try:
            c.drawImage(self.original_image_path, 50,
                        y - 200, width=200, height=200)
        except Exception as e:
            c.drawString(50, y, f"Error loading original image: {e}")
        y -= 210

        # Insert processed image
        c.drawString(50, y, "Processed Image:")
        y -= 15
        try:
            c.drawImage(self.processed_image_path, 50,
                        y - 200, width=200, height=200)
        except Exception as e:
            c.drawString(50, y, f"Error loading processed image: {e}")
        y -= 210

        if y < 100:
            c.showPage()
            y = height - 50

        # Now display the color ranges
        for idx, color_range in enumerate(self.color_ranges):
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, f"Color Range {idx + 1}:")
            y -= 15
            for color in color_range:
                # Convert BGR to RGB if necessary
                if isinstance(color, tuple) and len(color) == 3:
                    color_rgb = (color[2], color[1], color[0])
                else:
                    color_rgb = color
                hex_color = '#%02x%02x%02x' % color_rgb
                r, g, b = [value / 255 for value in color_rgb]
                c.setFillColor(Color(r, g, b))
                c.rect(60, y, 50, 10, fill=1, stroke=0)
                c.setFillColorRGB(0, 0, 0)
                c.setFont("Helvetica", 10)
                c.drawString(115, y + 2, hex_color)
                y -= 15
                if y < 50:
                    c.showPage()
                    y = height - 50
            y -= 10
            if y < 50:
                c.showPage()
                y = height - 50

        c.save()
