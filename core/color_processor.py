import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin_min
from collections import Counter
from typing import List, Tuple, Dict
from PIL import Image


class ColorProcessor:
    def __init__(self, image: Image.Image):
        """
        Initialize the ColorProcessor with an image.
        :param image: PIL.Image.Image - Input image to process.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL Image")

        # Convert PIL Image to OpenCV image format (numpy array)
        self.image_pil = image.convert('RGB')
        self.image = cv2.cvtColor(np.array(self.image_pil), cv2.COLOR_RGB2BGR)

        # Configuration parameters
        self.max_colors = 120  # Maximum number of colors in the final image
        self.min_colors = 80   # Minimum number of colors when appropriate
        # Minimum frequency for a color to be considered important
        self.min_color_frequency = 20
        self.max_groups = 12  # Maximum number of color groups
        self.min_groups = 8   # Minimum number of color groups
        # Allowed number of colors per group
        self.colors_per_group_options = [10, 12]
        # Weight to give critical regions during quantization
        self.critical_region_weight = 2.0

        # Variables to store processing results
        self.color_palette: List[Tuple[int, int, int]] = []
        self.color_groups: Dict[int, List[Tuple[int, int, int]]] = {}
        self.color_ranges: List[List[Tuple[int, int, int]]] = []
        self.quantized_image: np.ndarray = None

    def process_colors(self, progress_callback=None):
        """
        Process the image colors according to the specified requirements.
        :param progress_callback: Optional callback function to report progress.
        """
        steps = [
            (self.preprocess_image, 5),
            (self.extract_colors, 10),
            (self.detect_critical_regions, 20),
            (self.quantize_colors, 60),
            (self.group_colors, 80),
            (self.replace_colors, 90),
            (self.update_image, 100)
        ]

        for step, progress in steps:
            step()
            if progress_callback:
                progress_callback(progress)

    def preprocess_image(self):
        """
        Preprocess the image: resize if necessary, and convert color spaces.
        """
        # Optional: Resize the image if it's too large to reduce computation time
        max_dimension = 800  # You can adjust this value
        height, width = self.image.shape[:2]
        if max(height, width) > max_dimension:
            scaling_factor = max_dimension / max(height, width)
            self.image = cv2.resize(self.image, None, fx=scaling_factor, fy=scaling_factor,
                                    interpolation=cv2.INTER_AREA)
            print(
                f"Image resized to {self.image.shape[1]}x{self.image.shape[0]} for processing.")

    def extract_colors(self):
        """
        Extract colors from the image and calculate their frequencies.
        """
        # Reshape the image to a 2D array of pixels
        self.pixels = self.image.reshape((-1, 3))

        # Calculate color frequencies
        pixels_list = [tuple(pixel) for pixel in self.pixels]
        self.color_frequencies = Counter(pixels_list)
        print(
            f"Extracted {len(self.color_frequencies)} unique colors from the image.")

    def detect_critical_regions(self):
        """
        Detect critical regions like skin tones and faces.
        """
        # Create masks for critical regions
        self.skin_mask = self.detect_skin_regions()
        self.face_mask = self.detect_face_regions()
        self.critical_mask = cv2.bitwise_or(self.skin_mask, self.face_mask)
        print("Critical regions (skin and faces) detected.")

    def detect_skin_regions(self) -> np.ndarray:
        """
        Detect skin regions in the image using HSV color thresholds.
        :return: A binary mask of skin regions.
        """
        # Convert image to HSV color space
        image_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        # Define skin color range in HSV
        lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin = np.array([20, 150, 255], dtype=np.uint8)
        # Create a mask for skin regions
        skin_mask = cv2.inRange(image_hsv, lower_skin, upper_skin)
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        return skin_mask

    def detect_face_regions(self) -> np.ndarray:
        """
        Detect face regions in the image using Haar cascades.
        :return: A binary mask of face regions.
        """
        # Load pre-trained Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Convert image to grayscale for face detection
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5)
        # Create a mask for face regions
        face_mask = np.zeros_like(gray_image, dtype=np.uint8)
        for (x, y, w, h) in faces:
            cv2.rectangle(face_mask, (x, y), (x + w, y + h), 255, -1)
        return face_mask

    def quantize_colors(self):
        """
        Quantize colors in the image, treating critical and non-critical regions differently.
        """
        # Flatten the image and masks
        pixels = self.pixels
        critical_mask_flat = self.critical_mask.flatten()

        # Separate critical and non-critical pixels
        critical_pixels = pixels[critical_mask_flat > 0]
        non_critical_pixels = pixels[critical_mask_flat == 0]

        # Determine the total number of colors based on image complexity
        total_unique_colors = len(self.color_frequencies)
        if total_unique_colors < self.min_colors:
            self.max_colors = total_unique_colors
        else:
            self.max_colors = min(self.max_colors, total_unique_colors)
            self.max_colors = max(self.max_colors, self.min_colors)

        print(f"Total colors to quantize: {self.max_colors}")

        # Determine the number of colors for each region
        total_pixels = len(pixels)
        num_critical_pixels = len(critical_pixels)
        num_non_critical_pixels = len(non_critical_pixels)

        # Allocate colors based on pixel counts and critical region weight
        total_weight = num_critical_pixels * \
            self.critical_region_weight + num_non_critical_pixels
        num_critical_colors = int(
            (num_critical_pixels * self.critical_region_weight / total_weight) * self.max_colors)
        num_non_critical_colors = self.max_colors - num_critical_colors

        # Ensure the number of colors per group matches allowed options
        num_critical_colors = self._adjust_color_count(num_critical_colors)
        num_non_critical_colors = self.max_colors - num_critical_colors

        print(
            f"Quantizing critical regions into {num_critical_colors} colors.")
        print(
            f"Quantizing non-critical regions into {num_non_critical_colors} colors.")

        # Combine critical and non-critical pixels for clustering to merge similar colors
        all_pixels = np.vstack((critical_pixels, non_critical_pixels))
        num_clusters = num_critical_colors + num_non_critical_colors

        # Perform K-Means clustering on all pixels
        all_pixels_float = np.float32(all_pixels)
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42)
        all_labels = kmeans.fit_predict(all_pixels_float)
        centers = np.uint8(kmeans.cluster_centers_)

        # Map labels back to original pixels
        self.color_palette = centers
        self.labels = np.zeros(len(pixels), dtype=np.uint16)
        self.labels[critical_mask_flat > 0] = all_labels[:len(critical_pixels)]
        self.labels[critical_mask_flat ==
                    0] = all_labels[len(critical_pixels):]

        print(
            f"Total unique colors after quantization: {len(self.color_palette)}")

    def _adjust_color_count(self, color_count):
        # Adjust color count to be multiple of allowed colors per group
        colors_per_group = min(self.colors_per_group_options)
        adjusted_color_count = (
            color_count // colors_per_group) * colors_per_group
        if adjusted_color_count < colors_per_group:
            adjusted_color_count = colors_per_group
        return adjusted_color_count

    def group_colors(self):
        """
        Group colors into natural families and ensure light-to-warm progression.
        """
        # Convert color palette to Lab color space for perceptual uniformity
        palette_lab = cv2.cvtColor(
            self.color_palette.reshape(-1, 1, 3), cv2.COLOR_BGR2Lab).reshape(-1, 3)

        # Determine the number of groups based on allowed options
        total_colors = len(self.color_palette)
        for colors_per_group in self.colors_per_group_options:
            if total_colors % colors_per_group == 0:
                num_groups = total_colors // colors_per_group
                break
        else:
            # Default to minimum groups
            num_groups = self.min_groups

        # Perform Agglomerative Clustering to group colors
        clustering = AgglomerativeClustering(n_clusters=num_groups)
        group_labels = clustering.fit_predict(palette_lab)

        # Organize colors into groups
        self.color_groups = {i: [] for i in range(num_groups)}
        for color, group_label in zip(self.color_palette, group_labels):
            self.color_groups[group_label].append(tuple(int(c) for c in color))

        # Ensure each group has exact colors_per_group
        self._ensure_group_sizes()

        # Sort colors within each group
        for group_id, colors in self.color_groups.items():
            sorted_colors = self.sort_colors_within_group(colors)
            self.color_groups[group_id] = sorted_colors

        # Sort groups from lighter to warmer
        self._sort_groups()

        print(f"Colors grouped into {num_groups} natural families.")

    def _ensure_group_sizes(self):
        # Adjust groups to ensure each group has exactly colors_per_group colors
        colors_per_group = min(self.colors_per_group_options)
        all_colors = [color for group in self.color_groups.values()
                      for color in group]
        num_groups = len(self.color_groups)

        # Reassign colors to groups
        self.color_groups = {i: [] for i in range(num_groups)}
        for idx, color in enumerate(all_colors):
            group_id = idx // colors_per_group
            self.color_groups[group_id].append(color)

    def _sort_groups(self):
        """
        Sort the groups from lighter to warmer colors.
        """
        group_representatives = []
        for group_id, colors in self.color_groups.items():
            # Get the average color of the group in Lab space
            colors_np = np.array(colors, dtype=np.uint8).reshape(-1, 1, 3)
            colors_lab = cv2.cvtColor(
                colors_np, cv2.COLOR_BGR2Lab).reshape(-1, 3)
            avg_lab = np.mean(colors_lab, axis=0)
            group_representatives.append((group_id, avg_lab))

        # Sort groups based on lightness (L channel)
        sorted_groups = sorted(group_representatives,
                               key=lambda x: x[1][0], reverse=True)

        # Reorder self.color_groups based on sorted group IDs
        sorted_color_groups = {}
        for idx, (group_id, _) in enumerate(sorted_groups):
            sorted_color_groups[idx] = self.color_groups[group_id]
        self.color_groups = sorted_color_groups

    def sort_colors_within_group(self, colors: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """
        Sort colors within a group based on lightness and warmth.
        :param colors: List of colors in BGR format.
        :return: Sorted list of colors.
        """
        # Convert to Lab and HSV color spaces
        colors_np = np.array(colors, dtype=np.uint8).reshape(-1, 1, 3)
        colors_lab = cv2.cvtColor(colors_np, cv2.COLOR_BGR2Lab).reshape(-1, 3)
        sorting_keys = [lab[0] for lab in colors_lab]
        sorted_colors_with_keys = sorted(
            zip(sorting_keys, colors), reverse=True)
        sorted_colors = [color for _, color in sorted_colors_with_keys]
        return sorted_colors

    def replace_colors(self):
        """
        Replace colors in the image with the reduced palette, preserving critical colors.
        """
        # Map each label to its corresponding color in the palette
        new_pixels = self.color_palette[self.labels]
        self.quantized_image = new_pixels.reshape(self.image.shape)
        print("Colors replaced in the image with the reduced palette.")

    def update_image(self):
        """
        Update the image with the quantized colors and prepare color ranges.
        """
        # Convert the image back to PIL Image format
        self.image = Image.fromarray(cv2.cvtColor(
            self.quantized_image, cv2.COLOR_BGR2RGB))
        self.prepare_color_ranges()
        print("Image updated with quantized colors.")

    def prepare_color_ranges(self):
        """
        Prepare color ranges sorted by luminance and grouped naturally.
        """
        self.color_ranges = []
        for group_id in sorted(self.color_groups.keys()):
            colors = self.color_groups[group_id]
            self.color_ranges.append(colors)
        print("Color ranges prepared and sorted.")

    def get_image(self) -> Image.Image:
        """
        Get the processed image.
        :return: PIL.Image.Image - The quantized image.
        """
        return self.image

    def get_color_palette(self) -> List[Tuple[int, int, int]]:
        """
        Get the color palette used in the quantized image.
        :return: List of BGR color tuples.
        """
        return [tuple(int(c) for c in color) for color in self.color_palette]

    def get_color_ranges(self) -> List[List[Tuple[int, int, int]]]:
        """
        Get the color ranges grouped and sorted.
        :return: List of color groups, each containing a list of BGR color tuples.
        """
        return self.color_ranges
