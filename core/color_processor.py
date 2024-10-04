from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import math


class ColorProcessor:
    def __init__(self, image):
        self.image = image
        self.color_ranges = []
        self.max_colors = 120

    def process_colors(self, progress_callback=None):
        # Step 1: Extract colors
        if progress_callback:
            progress_callback.emit(10)
        self.extract_colors()

        # Step 2: Limit colors
        if progress_callback:
            progress_callback.emit(30)
        self.limit_colors()

        # Step 3: Group colors
        if progress_callback:
            progress_callback.emit(50)
        self.group_colors()

        # Step 4: Replace excess colors
        if progress_callback:
            progress_callback.emit(70)
        self.replace_excess_colors()

        if progress_callback:
            progress_callback.emit(100)

    def extract_colors(self):
        self.image = self.image.convert('RGB')
        self.image_data = np.array(self.image)
        self.pixels = self.image_data.reshape(-1, 3)

    def limit_colors(self):
        # Use KMeans clustering to reduce colors
        num_colors = min(self.max_colors, len(np.unique(self.pixels, axis=0)))
        kmeans = KMeans(n_clusters=num_colors, random_state=42)
        kmeans.fit(self.pixels)
        self.limited_colors = kmeans.cluster_centers_.astype(int)
        self.labels = kmeans.labels_

    def group_colors(self):
        # Group colors into ranges of 10 to 12, maximum of 10 ranges
        colors = self.limited_colors.tolist()
        group_size = min(max(10, math.ceil(len(colors)/10)), 12)
        self.color_ranges = [colors[i:i+group_size]
                             for i in range(0, len(colors), group_size)]
        self.color_ranges = self.color_ranges[:10]

    def replace_excess_colors(self):
        # Create a flat list of allowed colors
        allowed_colors = [tuple(color)
                          for group in self.color_ranges for color in group]

        # Map each pixel's color to the closest allowed color
        color_map = {}
        for idx, color in enumerate(self.limited_colors):
            original_color = tuple(color)
            if original_color in allowed_colors:
                color_map[idx] = original_color
            else:
                # Find the closest allowed color
                closest_color = min(
                    allowed_colors,
                    key=lambda x: np.linalg.norm(np.array(x) - color)
                )
                color_map[idx] = closest_color

        # Replace colors in the image data
        new_pixels = np.array([color_map[label] for label in self.labels])
        self.image_data = new_pixels.reshape(self.image_data.shape)

    def update_image(self):
        self.image = Image.fromarray(self.image_data.astype('uint8'), 'RGB')
