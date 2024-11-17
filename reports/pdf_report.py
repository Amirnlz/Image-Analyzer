from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color, HexColor
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from PIL import Image


class PDFReport:
    def __init__(self, color_ranges, original_image_path, processed_image_path):
        self.color_ranges = color_ranges
        self.original_image_path = original_image_path
        self.processed_image_path = processed_image_path

    def generate(self, filename="color_report.pdf"):
        c = canvas.Canvas(filename, pagesize=letter)
        page_width, page_height = letter
        margin = 50
        y_position = page_height - margin
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, y_position, "Color Report")
        y_position -= 30

        # Load images
        original_image = Image.open(self.original_image_path)
        processed_image = Image.open(self.processed_image_path)

        # Determine image placement based on orientation
        orig_width, orig_height = original_image.size
        proc_width, proc_height = processed_image.size

        orig_aspect = orig_width / orig_height
        proc_aspect = proc_width / proc_height

        # Calculate maximum dimensions for images
        max_image_width = page_width - 2 * margin
        available_height = y_position - margin - 50  # Leave space at the bottom
        # Half of available height minus some spacing
        max_image_height = available_height / 2 - 20

        # Resize images to fit within the page if necessary
        def get_scaled_dimensions(img_width, img_height, max_width, max_height):
            aspect_ratio = img_width / img_height
            if img_width > max_width or img_height > max_height:
                scaling_factor = min(max_width / img_width,
                                     max_height / img_height)
                return img_width * scaling_factor, img_height * scaling_factor
            else:
                return img_width, img_height

        orig_display_width, orig_display_height = get_scaled_dimensions(
            orig_width, orig_height, max_image_width, max_image_height)
        proc_display_width, proc_display_height = get_scaled_dimensions(
            proc_width, proc_height, max_image_width, max_image_height)

        # Draw original image
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y_position, "Original Image:")
        y_position -= 15
        c.drawImage(self.original_image_path, margin, y_position -
                    orig_display_height, width=orig_display_width, height=orig_display_height)
        # Adjust y_position after drawing the image
        y_position -= orig_display_height + 20

        # Draw processed image
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y_position, "Processed Image:")
        y_position -= 15
        c.drawImage(self.processed_image_path, margin, y_position -
                    proc_display_height, width=proc_display_width, height=proc_display_height)
        # Adjust y_position after drawing the image
        y_position -= proc_display_height + 20

        # Start a new page if necessary
        if y_position < margin + 100:
            c.showPage()
            y_position = page_height - margin

        # Now display the color ranges
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y_position, "Color Ranges:")
        y_position -= 20

        square_size = 40  # Size of the color squares
        padding = 10  # Padding between squares
        text_height = 12  # Height reserved for text
        max_squares_per_row = int(
            (page_width - 2 * margin) / (square_size + padding))

        for idx, color_range in enumerate(self.color_ranges):
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, y_position, f"Group {idx + 1}:")
            y_position -= square_size + text_height + 10  # Space for the squares and text

            x_position = margin
            count = 0
            for color in color_range:
                # Convert BGR to RGB if necessary
                if isinstance(color, tuple) and len(color) == 3:
                    color_rgb = (color[2], color[1], color[0])
                else:
                    color_rgb = color

                hex_color = '#%02x%02x%02x' % color_rgb
                r, g, b = [value / 255 for value in color_rgb]
                c.setFillColor(Color(r, g, b))
                c.rect(x_position, y_position - text_height,
                       square_size, square_size, fill=1, stroke=0)

                # Lighter color for text
                text_color = Color(1 - r * 0.5, 1 - g * 0.5, 1 - b * 0.5)
                c.setFillColor(text_color)
                c.setFont("Helvetica", 8)
                c.drawCentredString(
                    x_position + square_size / 2, y_position - text_height - 10, hex_color)

                x_position += square_size + padding
                count += 1

                if count % max_squares_per_row == 0:
                    x_position = margin
                    y_position -= square_size + text_height + padding

                    if y_position < margin + square_size + text_height + 50:
                        c.showPage()
                        y_position = page_height - margin

            y_position -= square_size + text_height + 20  # Space between groups

            if y_position < margin + square_size + text_height + 50:
                c.showPage()
                y_position = page_height - margin

        c.save()
