from datetime import datetime
import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QLineEdit,
    QFileDialog, QMessageBox, QVBoxLayout, QHBoxLayout, QProgressBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from core.calculator import Calculator
from core.color_processor import ColorProcessor
from core.image_handler import ImageHandler
from reports.pdf_report import PDFReport


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processor")
        self.image_handler = ImageHandler()
        self.calculator = Calculator()
        self.color_processor = None
        self.current_aspect_ratio = 1.0
        self.setup_ui()
        self.width_cm = None
        self.height_cm = None

    def setup_ui(self):
        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout()
        self.main_widget.setLayout(self.layout)

        # Image Selection
        self.select_button = QPushButton("Select Image")
        self.select_button.clicked.connect(self.select_image)
        self.layout.addWidget(self.select_button)

        # Density Input
        density_layout = QHBoxLayout()
        self.density_label = QLabel("Density:")
        self.density_input = QLineEdit("45")
        self.density_input.textChanged.connect(self.update_calculation)
        density_layout.addWidget(self.density_label)
        density_layout.addWidget(self.density_input)
        self.layout.addLayout(density_layout)

        # Size Display
        self.size_label = QLabel("Image Size (cm):")
        self.layout.addWidget(self.size_label)
        self.width_label = QLabel("Width: N/A")
        self.height_label = QLabel("Height: N/A")
        self.layout.addWidget(self.width_label)
        self.layout.addWidget(self.height_label)

        # Size Adjustment
        adjust_layout = QHBoxLayout()
        self.adjust_label = QLabel("Adjust Width (cm):")
        self.width_input = QLineEdit()
        self.width_input.textChanged.connect(self.adjust_size)
        adjust_layout.addWidget(self.adjust_label)
        adjust_layout.addWidget(self.width_input)
        self.layout.addLayout(adjust_layout)

        # Process Button
        self.process_button = QPushButton("Process Image")
        self.process_button.clicked.connect(self.process_image)
        self.layout.addWidget(self.process_button)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if file_path:
            self.image_handler.open_image(file_path)
            self.calculate_size()

    def calculate_size(self):
        try:
            density = float(self.density_input.text())
            self.calculator.density = density
        except ValueError:
            QMessageBox.warning(self, "Invalid Input",
                                "Please enter a valid density.")
            return

        self.width_cm, self.height_cm = self.calculator.calculate_size(
            self.image_handler.width_px,
            self.image_handler.height_px
        )

        self.width_label.setText(f"Width: {self.width_cm:.2f} cm")
        self.height_label.setText(f"Height: {self.height_cm:.2f} cm")
        self.width_input.setText(f"{self.width_cm:.2f}")
        self.current_aspect_ratio = self.height_cm / self.width_cm

    def update_calculation(self):
        if self.image_handler.image:
            self.calculate_size()

    def adjust_size(self):
        try:
            new_width_cm = float(self.width_input.text())
        except ValueError:
            return

        new_height_cm = new_width_cm * self.current_aspect_ratio
        self.width_label.setText(f"Width: {new_width_cm:.2f} cm")
        self.height_label.setText(f"Height: {new_height_cm:.2f} cm")

        # Update the stored width and height
        self.width_cm = new_width_cm
        self.height_cm = new_height_cm

    def process_image(self):
        if not self.image_handler.image:
            QMessageBox.warning(
                self, "No Image", "Please select an image first.")
            return

        # Start processing in a separate thread
        self.thread = QThread()
        self.worker = ProcessingWorker(self.image_handler.image)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.result.connect(self.processing_finished)

        self.thread.start()
        self.process_button.setEnabled(False)
        self.thread.finished.connect(
            lambda: self.process_button.setEnabled(True))

    def processing_finished(self, color_processor):
        self.color_processor = color_processor
        self.color_processor.update_image()
        self.image_handler.image = self.color_processor.image

        # Prepare variables for filename
        width_cm = self.width_cm
        height_cm = self.height_cm
        density = self.calculator.density

        # Create the report directory if it doesn't exist
        report_dir = os.path.join(os.getcwd(), 'report')
        os.makedirs(report_dir, exist_ok=True)

        # Construct the filename with width, height, and density
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"processed_image_{width_cm:.2f}x{height_cm:.2f}_density{density}_{timestamp}.png"
        image_path = os.path.join(report_dir, image_filename)

        # Save the image
        self.image_handler.save_image(image_path)

        # Generate PDF report
        pdf_filename = f"color_report_{timestamp}.pdf"
        pdf_path = os.path.join(report_dir, pdf_filename)
        report = PDFReport(self.color_processor.color_ranges)
        report.generate(pdf_path)

        QMessageBox.information(
            self,
            "Success",
            f"Image processed and saved to:\n{image_path}\n\nPDF report saved to:\n{pdf_path}"
        )


class ProcessingWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    result = pyqtSignal(object)

    def __init__(self, image):
        super().__init__()
        self.image = image

    def run(self):
        processor = ColorProcessor(self.image)
        processor.process_colors(self.progress)
        self.result.emit(processor)
        self.finished.emit()
