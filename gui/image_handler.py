from PIL import Image
from tkinter import filedialog

class ImageHandler:
    def __init__(self):
        self.image = None
        self.width_px = 0
        self.height_px = 0

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.png;*.jpeg;*.bmp;*.gif")]
        )
        if file_path:
            self.image = Image.open(file_path)
            self.width_px, self.height_px = self.image.size

    def save_image(self, path):
        if self.image:
            self.image.save(path)
