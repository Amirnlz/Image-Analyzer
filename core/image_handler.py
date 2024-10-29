import os 
from PIL import Image

class ImageHandler:
    def __init__(self):
        self.image = None
        self.original_image = None
        self.width_px = 0
        self.height_px = 0

    def open_image(self, file_path):
        self.image = Image.open(file_path)
        self.original_image = self.image.copy()  # Store a copy of the original image
        self.width_px, self.height_px = self.image.size

    def save_image(self, path):
        if self.image is not None:
            ext = os.path.splitext(path)[1].lower()
            if ext == '.png':
                image_format = 'PNG'
            elif ext in ['.jpg', '.jpeg']:
                image_format = 'JPEG'
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
            self.image.save(path, format=image_format)
        else:
            raise ValueError("No image to save.")

    def save_original_image(self, path):
        if self.original_image is not None:
            ext = os.path.splitext(path)[1].lower()
            if ext == '.png':
                image_format = 'PNG'
            elif ext in ['.jpg', '.jpeg']:
                image_format = 'JPEG'
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
            self.original_image.save(path, format=image_format)
        else:
            raise ValueError("No original image to save.")
