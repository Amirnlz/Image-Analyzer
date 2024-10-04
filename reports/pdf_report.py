from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color


class PDFReport:
    def __init__(self, color_ranges):
        self.color_ranges = color_ranges

    def generate(self, filename="color_report.pdf"):
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter
        y = height - 50
        c.setFont("Helvetica", 12)
        c.drawString(50, y, "Color Report")
        y -= 20

        for idx, color_range in enumerate(self.color_ranges):
            c.drawString(50, y, f"Color Range {idx + 1}:")
            y -= 15
            for color in color_range:
                hex_color = '#%02x%02x%02x' % tuple(color)
                r, g, b = [value / 255 for value in color]
                c.setFillColor(Color(r, g, b))
                c.rect(60, y, 50, 10, fill=1)
                c.setFillColorRGB(0, 0, 0)
                c.drawString(115, y + 2, hex_color)
                y -= 15
                if y < 50:
                    c.showPage()
                    y = height - 50
        c.save()
