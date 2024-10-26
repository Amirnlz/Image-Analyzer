from PIL import Image
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Callable, Optional, Union
from skimage import color
from collections import Counter


class ColorProcessor:
    def __init__(self, image: Image.Image):
        """Initialize the color processor with an image"""
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL Image")

        self.image = image
        self.color_ranges: List[List[Tuple[int, int, int]]] = []
        self.max_colors_per_category = 12
        self.max_categories = 10
        self.min_color_frequency = 20
        self.skin_tone_tolerance = 0.1

    def process_colors(self, progress_callback: Optional[Callable] = None) -> None:
        """Process colors with progress tracking"""
        try:
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
                try:
                    step()
                    if progress_callback:
                        progress_callback(progress)
                except Exception as e:
                    raise RuntimeError(
                        f"Error in step {step.__name__}: {str(e)}")

        except Exception as e:
            raise RuntimeError(f"Color processing failed: {str(e)}")

    def extract_colors(self) -> None:
        """Extract colors and convert to different color spaces"""
        try:
            self.image = self.image.convert('RGB')
            self.image_data = np.array(self.image)
            self.pixels_rgb = self.image_data.reshape(-1, 3)

            # Normalize RGB values to [0, 1] range
            pixels_rgb_norm = self.pixels_rgb.astype(float) / 255.0

            # Convert to different color spaces
            self.pixels_lab = color.rgb2lab(
                pixels_rgb_norm.reshape(1, -1, 3))[0]
            self.pixels_hsv = color.rgb2hsv(
                pixels_rgb_norm.reshape(1, -1, 3))[0]

            # Calculate color frequencies
            self.color_frequencies = Counter(map(tuple, self.pixels_rgb))
        except Exception as e:
            raise RuntimeError(f"Color extraction failed: {str(e)}")

    def analyze_color_distribution(self) -> None:
        """Analyze color distribution and identify important colors"""
        try:
            self.sorted_colors = sorted(
                self.color_frequencies.items(),
                key=lambda x: x[1],
                reverse=True
            )

            total_pixels = len(self.pixels_rgb)
            self.important_colors = {
                color for color, freq in self.sorted_colors
                if freq >= self.min_color_frequency
            }
        except Exception as e:
            raise RuntimeError(f"Color distribution analysis failed: {str(e)}")

    def detect_special_cases(self) -> None:
        """Detect special cases like skin tones"""
        try:
            self.special_cases = {
                'skin_tones': set(),
                'dominant_colors': set(),
                'edge_colors': set()
            }

            total_pixels = len(self.pixels_rgb)
            for color, freq in self.sorted_colors:
                # Convert color to numpy array for processing
                color_array = np.array(color, dtype=float)
                if self.is_skin_tone(color_array):
                    self.special_cases['skin_tones'].add(color)
                if freq > total_pixels * 0.01:  # More than 1% of image
                    self.special_cases['dominant_colors'].add(color)
        except Exception as e:
            raise RuntimeError(f"Special case detection failed: {str(e)}")

    def is_skin_tone(self, rgb: np.ndarray) -> bool:
        """Detect if a color is likely to be a skin tone"""
        try:
            # Normalize RGB values to [0, 1]
            rgb_norm = rgb / 255.0

            # Reshape for skimage color conversion
            rgb_reshaped = rgb_norm.reshape(1, 1, 3)

            # Convert to HSV
            hsv = color.rgb2hsv(rgb_reshaped)[0][0]
            h, s, v = hsv

            # Skin tone typically has these characteristics
            hue_ok = (h >= 0.0 and h <= 0.1) or (h >= 0.9 and h <= 1.0)
            sat_ok = s >= 0.2 and s <= 0.6
            val_ok = v >= 0.4 and v <= 1.0

            return hue_ok and sat_ok and val_ok
        except Exception as e:
            raise RuntimeError(f"Skin tone detection failed: {str(e)}")

    def group_colors(self) -> None:
        """Group colors using multiple clustering approaches"""
        try:
            # Normalize LAB colors for clustering
            scaler = StandardScaler()
            lab_normalized = scaler.fit_transform(self.pixels_lab)

            # Initial clustering with DBSCAN
            dbscan = DBSCAN(eps=0.3, min_samples=5)
            rough_labels = dbscan.fit_predict(lab_normalized)

            # Handle case where DBSCAN finds no clusters
            unique_labels = np.unique(rough_labels[rough_labels >= 0])
            if len(unique_labels) == 0:
                num_categories = self.max_categories
            else:
                num_categories = min(len(unique_labels), self.max_categories)

            # Refine with KMeans
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
        except Exception as e:
            raise RuntimeError(f"Color grouping failed: {str(e)}")

    def find_similar_color(self, color: np.ndarray,
                           palette: List[Tuple[int, int, int]],
                           is_special: bool = False) -> Tuple[int, int, int]:
        """Find the most similar color in the palette"""
        try:
            # Convert single color and palette to LAB space
            color_norm = color.reshape(1, 1, 3) / 255.0
            color_lab = color.rgb2lab(color_norm)[0][0]

            palette_array = np.array(
                palette, dtype=float).reshape(-1, 1, 3) / 255.0
            palette_lab = color.rgb2lab(palette_array)[:, 0, :]

            # Calculate color differences
            differences = np.array([
                color.delta_e(color_lab, p_lab, method='ciede2000')
                for p_lab in palette_lab
            ])

            # For special cases, use stricter similarity threshold
            if is_special:
                threshold = 5.0
                valid_indices = differences < threshold
                if np.any(valid_indices):
                    differences = differences[valid_indices]
                    palette = np.array(palette)[valid_indices]

            closest_color = palette[np.argmin(differences)]
            return tuple(map(int, closest_color))
        except Exception as e:
            raise RuntimeError(
                f"Color similarity calculation failed: {str(e)}")

    def limit_colors_within_categories(self) -> None:
        """Limit colors within each category while preserving special cases"""
        try:
            self.limited_categories = {}
            self.color_palette = []

            for label, category in self.color_categories.items():
                pixels = category['pixels']
                special_cases = category['special_cases']

                # Ensure special cases are preserved
                preserved_colors = {color for _, color in special_cases}
                remaining_slots = self.max_colors_per_category - \
                    len(preserved_colors)

                if remaining_slots > 0 and pixels:
                    # Cluster remaining colors
                    remaining_pixels = [
                        p for p in pixels if p not in preserved_colors]
                    if remaining_pixels:
                        kmeans = KMeans(
                            n_clusters=min(remaining_slots,
                                           len(remaining_pixels)),
                            random_state=42
                        )
                        remaining_pixels_array = np.array(remaining_pixels)
                        labels = kmeans.fit_predict(remaining_pixels_array)
                        centers = kmeans.cluster_centers_

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
        except Exception as e:
            raise RuntimeError(f"Color limitation failed: {str(e)}")

    def replace_excess_colors(self) -> None:
        """Replace colors while preserving special cases"""
        try:
            new_pixels = np.zeros_like(self.pixels_rgb)

            for label, category in self.limited_categories.items():
                indices = category['indices']
                palette = category['palette']

                if not palette:  # Skip if palette is empty
                    continue

                # Process each pixel in the category
                for idx in indices:
                    original_color = self.pixels_rgb[idx]
                    original_tuple = tuple(original_color)
                    is_special = any(
                        original_tuple == color
                        for _, color in self.color_categories[label]['special_cases']
                    )

                    if original_tuple in palette:
                        new_pixels[idx] = original_color
                    else:
                        new_color = self.find_similar_color(
                            original_color, palette, is_special)
                        new_pixels[idx] = new_color

            self.image_data = new_pixels.reshape(self.image_data.shape)
        except Exception as e:
            raise RuntimeError(f"Color replacement failed: {str(e)}")

    def update_image(self) -> None:
        """Update image and prepare color ranges"""
        try:
            self.image = Image.fromarray(
                self.image_data.astype('uint8'), 'RGB')
            self.prepare_color_ranges()
        except Exception as e:
            raise RuntimeError(f"Image update failed: {str(e)}")

    def prepare_color_ranges(self) -> None:
        """Prepare color ranges sorted by luminance"""
        try:
            self.color_ranges = []
            for category in self.limited_categories.values():
                if not category['palette']:  # Skip empty palettes
                    continue

                # Convert to LAB for better luminance sorting
                palette_array = np.array(
                    category['palette'], dtype=float) / 255.0
                palette_lab = color.rgb2lab(
                    palette_array.reshape(-1, 1, 3))[:, 0, :]

                # Sort by L value (luminance)
                sorted_indices = np.argsort(
                    [lab[0] for lab in palette_lab])[::-1]
                sorted_colors = [category['palette'][i]
                                 for i in sorted_indices]
                self.color_ranges.append(sorted_colors)
        except Exception as e:
            raise RuntimeError(f"Color range preparation failed: {str(e)}")
