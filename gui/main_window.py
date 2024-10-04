import tkinter as tk
from core.image_handler import ImageHandler
from core.calculator import Calculator
from core.color_processor import ColorProcessor
from reports.pdf_report import PDFReport

class MainWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Image Processor")
        self.setup_gui()
        self.image_handler = ImageHandler()
        self.calculator = Calculator()
        self.color_processor = None

    def setup_gui(self):
        # Set up GUI components (buttons, entries, labels)
        # Image selection
        self.select_button = tk.Button(self.root, text="Select Image", command=self.select_image)
        self.select_button.pack()
        # Density input
        # Size display and adjustment
        # Process button
        self.process_button = tk.Button(self.root, text="Process Image", command=self.process_image)
        self.process_button.pack()

    def select_image(self):
        # Use image_handler to open image
        self.image_handler.open_image()
        # Update GUI with image details
        # Calculate and display image size
        self.update_image_info()

    def update_image_info(self):
        # Calculate size using calculator
        width_cm, height_cm = self.calculator.calculate_size(
            self.image_handler.width_px,
            self.image_handler.height_px,
            density=self.calculator.density
        )
        # Update GUI labels and entries
        pass

    def process_image(self):
        # Initialize color processor
        self.color_processor = ColorProcessor(self.image_handler.image)
        self.color_processor.process_colors()
        # Generate PDF report
        report = PDFReport(self.color_processor.color_ranges)
        report.generate()
        # Notify user
        tk.messagebox.showinfo("Success", "Image processed and PDF report generated.")

    def run(self):
        self.root.mainloop()
