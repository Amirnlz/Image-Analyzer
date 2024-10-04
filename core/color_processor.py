from PIL import Image
import numpy as np
import math
from sklearn.cluster import KMeans


class ColorProcessor:
    def __init__(self, image):
        self.image = image
        self.color_ranges = []
        self.max_colors_per_category = 12
        self.max_categories = 10
        self.hue_tolerance = 15  # Degrees of hue tolerance for grouping

    def process_colors(self, progress_callback=None):
        if progress_callback:
            progress_callback.emit(10)
        self.extract_colors()

        if progress_callback:
            progress_callback.emit(30)
        self.group_colors_by_hue()

        if progress_callback:
            progress_callback.emit(60)
        self.limit_colors_within_categories()

        if progress_callback:
            progress_callback.emit(80)
        self.replace_excess_colors()

        if progress_callback:
            progress_callback.emit(100)

    def extract_colors(self):
        self.image = self.image.convert('RGB')
        self.image_data = np.array(self.image)
        self.pixels = self.image_data.reshape(-1, 3)

        # Convert RGB pixels to HSV
        self.pixels_hsv = np.array([self.rgb_to_hsv(pixel)
                                   for pixel in self.pixels])

    def group_colors_by_hue(self):
        # Group colors based on hue
        hues = self.pixels_hsv[:, 0]
        num_categories = min(len(np.unique(hues)), self.max_categories)
        kmeans = KMeans(n_clusters=num_categories, random_state=42)
        kmeans.fit(hues.reshape(-1, 1))
        self.hue_labels = kmeans.labels_
        self.cluster_centers = kmeans.cluster_centers_

        # Create hue categories and store pixel indices
        self.hue_categories = {}
        for idx, label in enumerate(self.hue_labels):
            pixel_rgb = self.pixels[idx]
            pixel_hsv = self.pixels_hsv[idx]
            if label not in self.hue_categories:
                self.hue_categories[label] = {'pixels': [], 'indices': []}
            self.hue_categories[label]['pixels'].append(
                {'rgb': pixel_rgb, 'hsv': pixel_hsv})
            self.hue_categories[label]['indices'].append(idx)

    def limit_colors_within_categories(self):
        self.limited_categories = {}
        self.color_palette = []
        for label, category_data in self.hue_categories.items():
            hsv_values = np.array([pixel['hsv']
                                  for pixel in category_data['pixels']])
            num_pixels = len(hsv_values)
            num_colors = min(num_pixels, self.max_colors_per_category)

            # Cluster based on value (brightness)
            kmeans = KMeans(n_clusters=num_colors, random_state=42)
            kmeans.fit(hsv_values[:, 2].reshape(-1, 1))
            cluster_centers = kmeans.cluster_centers_
            labels = kmeans.labels_

            # Build the limited color palette for this category
            category_palette = []
            for center in cluster_centers:
                hsv = np.array([self.cluster_centers[label][0],
                               np.mean(hsv_values[:, 1]), center[0]])
                rgb = self.hsv_to_rgb(hsv)
                category_palette.append(rgb)

            self.limited_categories[label] = {
                'palette': category_palette,
                'labels': labels,
                'indices': category_data['indices']
            }
            self.color_palette.extend(category_palette)

    def replace_excess_colors(self):
        # Initialize an array for new pixels
        new_pixels = np.zeros_like(self.pixels)

        # Iterate over each hue category
        for label, category in self.limited_categories.items():
            indices = category['indices']
            cluster_labels = category['labels']
            palette = category['palette']

            # Replace colors for pixels in this category
            for idx_in_category, original_idx in enumerate(indices):
                cluster_label = cluster_labels[idx_in_category]
                new_color = palette[cluster_label]
                new_pixels[original_idx] = new_color

        self.image_data = new_pixels.reshape(self.image_data.shape)

    def update_image(self):
        self.image = Image.fromarray(self.image_data.astype('uint8'), 'RGB')
        # Prepare color ranges for PDF report
        self.prepare_color_ranges()

    def prepare_color_ranges(self):
        self.color_ranges = []
        for label in self.limited_categories:
            palette = self.limited_categories[label]['palette']
            # Sort palette from light to dark based on value
            palette_sorted = sorted(
                palette, key=lambda rgb: self.rgb_to_hsv(rgb)[2], reverse=True)
            self.color_ranges.append(palette_sorted)

    @staticmethod
    def rgb_to_hsv(rgb):
        r, g, b = rgb / 255.0
        maxc = max(r, g, b)
        minc = min(r, g, b)
        v = maxc
        if maxc == minc:
            h = 0.0
            s = 0.0
        else:
            delta = maxc - minc
            s = delta / maxc
            if r == maxc:
                h = (g - b) / delta
            elif g == maxc:
                h = 2.0 + (b - r) / delta
            else:
                h = 4.0 + (r - g) / delta
            h = (h * 60) % 360
        return np.array([h, s, v])

    @staticmethod
    def hsv_to_rgb(hsv):
        h, s, v = hsv
        h = h % 360
        c = v * s
        x = c * (1 - abs(((h / 60) % 2) - 1))
        m = v - c
        if h < 60:
            r1, g1, b1 = c, x, 0
        elif h < 120:
            r1, g1, b1 = x, c, 0
        elif h < 180:
            r1, g1, b1 = 0, c, x
        elif h < 240:
            r1, g1, b1 = 0, x, c
        elif h < 300:
            r1, g1, b1 = x, 0, c
        else:
            r1, g1, b1 = c, 0, x
        r, g, b = (r1 + m), (g1 + m), (b1 + m)
        return np.array([int(r * 255), int(g * 255), int(b * 255)])
