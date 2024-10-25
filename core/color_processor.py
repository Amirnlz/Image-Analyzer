from PIL import Image
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Callable, Optional, Union
from skimage import color
from collections import Counter


class ColorProcessor:
    def __init__(self, image: Image.Image):
        self.image = image
        self.color_ranges: List[List[Tuple[int, int, int]]] = []
        self.max_colors_per_category = 12
        self.max_categories = 10
        self.min_color_frequency = 20  # Minimum frequency to preserve a color
        self.skin_tone_tolerance = 0.1  # Tolerance for skin tone detection

    def process_colors(self, progress_callback: Optional[Callable] = None) -> None:
        steps = [
            (self.extract_colors, 10),
            (self.analyze_color_distribution, 20),
            (self.detect_special_cases, 30),
            (self.group_colors, 50),
            (self.limit_colors_within_categories, 70),
            (self.replace_excess_colors, 90),
            (self.update_image, 100)
        ]

        for step, progress in steps:
            step()
            if progress_callback:
                progress_callback(progress)

    def extract_colors(self) -> None:
        """Extract colors and convert to different color spaces"""
        self.image = self.image.convert('RGB')
        self.image_data = np.array(self.image)
        self.pixels_rgb = self.image_data.reshape(-1, 3)

        # Convert to different color spaces
        self.pixels_lab = color.rgb2lab(self.pixels_rgb / 255.0)
        self.pixels_hsv = color.rgb2hsv(self.pixels_rgb / 255.0)

        # Calculate color frequencies
        self.color_frequencies = Counter(map(tuple, self.pixels_rgb))

    def analyze_color_distribution(self) -> None:
        """Analyze color distribution and identify important colors"""
        # Sort colors by frequency
        self.sorted_colors = sorted(
            self.color_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Identify important colors (high frequency)
        self.important_colors = {
            color for color, freq in self.sorted_colors
            if freq >= self.min_color_frequency
        }

    def detect_special_cases(self) -> None:
        """Detect special cases like skin tones"""
        self.special_cases = {
            'skin_tones': set(),
            'dominant_colors': set(),
            'edge_colors': set()
        }

        for color, freq in self.sorted_colors:
            if self.is_skin_tone(color):
                self.special_cases['skin_tones'].add(color)
            if freq > len(self.pixels_rgb) * 0.01:  # More than 1% of image
                self.special_cases['dominant_colors'].add(color)

    def is_skin_tone(self, rgb: Tuple[int, int, int]) -> bool:
        """Detect if a color is likely to be a skin tone"""
        r, g, b = np.array(rgb) / 255.0
        h, s, v = color.rgb2hsv(r, g, b)

        # Skin tone typically has these characteristics
        hue_ok = (h >= 0.0 and h <= 0.1) or (h >= 0.9 and h <= 1.0)
        sat_ok = s >= 0.2 and s <= 0.6
        val_ok = v >= 0.4 and v <= 1.0

        return hue_ok and sat_ok and val_ok

    def group_colors(self) -> None:
        """Group colors using multiple clustering approaches"""
        # Normalize LAB colors for clustering
        scaler = StandardScaler()
        lab_normalized = scaler.fit_transform(self.pixels_lab)

        # Initial clustering with DBSCAN to find natural groups
        dbscan = DBSCAN(eps=0.3, min_samples=5)
        rough_labels = dbscan.fit_predict(lab_normalized)

        # Refine with KMeans
        num_categories = min(len(np.unique(rough_labels[rough_labels >= 0])),
                             self.max_categories)

        kmeans = KMeans(n_clusters=num_categories, random_state=42)
        self.color_labels = kmeans.fit_predict(lab_normalized)

        # Organize colors into categories
        self.color_categories = {
            label: {'pixels': [], 'indices': [], 'special_cases': set()}
            for label in range(num_categories)
        }

        for idx, (label, pixel_rgb) in enumerate(zip(self.color_labels, self.pixels_rgb)):
            pixel_tuple = tuple(pixel_rgb)
            category = self.color_categories[label]
            category['pixels'].append(pixel_tuple)
            category['indices'].append(idx)

            # Track special cases in each category
            if pixel_tuple in self.special_cases['skin_tones']:
                category['special_cases'].add(('skin_tone', pixel_tuple))
            if pixel_tuple in self.special_cases['dominant_colors']:
                category['special_cases'].add(('dominant', pixel_tuple))

    def find_similar_color(self, color: Tuple[int, int, int],
                           palette: List[Tuple[int, int, int]],
                           is_special: bool = False) -> Tuple[int, int, int]:
        """Find the most similar color in the palette"""
        color_lab = color.rgb2lab(np.array([[color]]) / 255.0)[0][0]
        palette_lab = color.rgb2lab(np.array([palette]) / 255.0)

        # Calculate color differences using CIEDE2000
        differences = np.array([
            color.delta_e(color_lab, p_lab, method='ciede2000')
            for p_lab in palette_lab
        ])

        # For special cases, use stricter similarity threshold
        if is_special:
            threshold = 5.0  # Smaller threshold for special cases
            valid_indices = differences < threshold
            if np.any(valid_indices):
                differences = differences[valid_indices]
                palette = np.array(palette)[valid_indices]

        return palette[np.argmin(differences)]

    def limit_colors_within_categories(self) -> None:
        """Limit colors within each category while preserving special cases"""
        self.limited_categories = {}
        self.color_palette = []

        for label, category in self.color_categories.items():
            pixels = category['pixels']
            special_cases = category['special_cases']

            # Ensure special cases are preserved
            preserved_colors = {color for _, color in special_cases}
            remaining_slots = self.max_colors_per_category - \
                len(preserved_colors)

            if remaining_slots > 0:
                # Cluster remaining colors
                remaining_pixels = [
                    p for p in pixels if p not in preserved_colors]
                if remaining_pixels:
                    kmeans = KMeans(
                        n_clusters=min(remaining_slots, len(remaining_pixels)),
                        random_state=42
                    )
                    remaining_pixels_array = np.array(remaining_pixels)
                    labels = kmeans.fit_predict(remaining_pixels_array)
                    centers = kmeans.cluster_centers_

                    # Add cluster centers to palette
                    category_palette = list(preserved_colors) + [
                        tuple(map(int, center)) for center in centers
                    ]
                else:
                    category_palette = list(preserved_colors)
            else:
                category_palette = list(preserved_colors)[
                    :self.max_colors_per_category]

            self.limited_categories[label] = {
                'palette': category_palette,
                'indices': category['indices']
            }
            self.color_palette.extend(category_palette)

    def replace_excess_colors(self) -> None:
        """Replace colors while preserving special cases"""
        new_pixels = np.zeros_like(self.pixels_rgb)

        for label, category in self.limited_categories.items():
            indices = category['indices']
            palette = category['palette']

            # Process each pixel in the category
            for idx in indices:
                original_color = tuple(self.pixels_rgb[idx])
                is_special = any(
                    original_color == color
                    for _, color in self.color_categories[label]['special_cases']
                )

                new_color = (
                    original_color if original_color in palette
                    else self.find_similar_color(original_color, palette, is_special)
                )
                new_pixels[idx] = new_color

        self.image_data = new_pixels.reshape(self.image_data.shape)

    def update_image(self) -> None:
        """Update image and prepare color ranges"""
        self.image = Image.fromarray(self.image_data.astype('uint8'), 'RGB')
        self.prepare_color_ranges()

    def prepare_color_ranges(self) -> None:
        """Prepare color ranges sorted by luminance"""
        self.color_ranges = []
        for category in self.limited_categories.values():
            # Convert to LAB for better luminance sorting
            palette_lab = color.rgb2lab(
                np.array([category['palette']]) / 255.0)[0]
            # Sort by L value (luminance)
            sorted_colors = [
                category['palette'][i]
                for i in np.argsort([lab[0] for lab in palette_lab])[::-1]
            ]
            self.color_ranges.append(sorted_colors)
