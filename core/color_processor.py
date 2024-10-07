from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple, Callable, Optional, Union


class ColorProcessor:
    def __init__(self, image: Image.Image, min_color_usage=20):
        self.image = image
        self.min_color_usage = min_color_usage
        self.color_ranges: List[List[Tuple[int, int, int]]] = []
        self.max_colors_per_category = 12
        self.max_categories = 10
        self.hue_tolerance = 15  # Degrees of hue tolerance for grouping
        self.color_usage = {}

    def process_colors(self, progress_callback: Optional[Callable] = None) -> None:
        steps = [
            (self.extract_colors, 10),
            (self.group_colors_by_hue, 30),
            (self.limit_colors_within_categories, 60),
            (self.replace_excess_colors, 80),
            (self.update_image, 100)
        ]

        for step, progress in steps:
            step()
            if progress_callback:
                progress_callback(progress)

    def extract_colors(self) -> None:
        self.image = self.image.convert('RGB')
        self.image_data = np.array(self.image)
        self.pixels = self.image_data.reshape(-1, 3)
        self.pixels_hsv = np.array([self.rgb_to_hsv(pixel) for pixel in self.pixels])

    def group_colors_by_hue(self) -> None:
        hues = self.pixels_hsv[:, 0]
        num_categories = min(len(np.unique(hues)), self.max_categories)
        kmeans = KMeans(n_clusters=num_categories, random_state=42)
        self.hue_labels = kmeans.fit_predict(hues.reshape(-1, 1))
        self.cluster_centers = kmeans.cluster_centers_

        self.hue_categories: Dict[int, Dict] = {
            label: {'pixels': [], 'indices': []}
            for label in range(num_categories)
        }

        for idx, (label, pixel_rgb, pixel_hsv) in enumerate(zip(self.hue_labels, self.pixels, self.pixels_hsv)):
            self.hue_categories[label]['pixels'].append({'rgb': pixel_rgb, 'hsv': pixel_hsv})
            self.hue_categories[label]['indices'].append(idx)

    def limit_colors_within_categories(self) -> None:
        self.limited_categories: Dict[int, Dict] = {}
        self.color_palette: List[Tuple[int, int, int]] = []
        self.color_usage: Dict[Tuple[int, int, int], int] = {}

        for label, category_data in self.hue_categories.items():
            hsv_values = np.array([pixel['hsv'] for pixel in category_data['pixels']])
            num_colors = min(len(hsv_values), self.max_colors_per_category)

            kmeans = KMeans(n_clusters=num_colors, random_state=42)
            labels = kmeans.fit_predict(hsv_values[:, 2].reshape(-1, 1))
            cluster_centers = kmeans.cluster_centers_

            category_palette = [
                self.hsv_to_rgb(np.array([self.cluster_centers[label][0], np.mean(hsv_values[:, 1]), center[0]]))
                for center in cluster_centers
            ]

            # Count how many times each color appears
            for idx, cluster_label in enumerate(labels):
                rgb_color = tuple(category_palette[cluster_label])
                self.color_usage[rgb_color] = self.color_usage.get(rgb_color, 0) + 1

            self.limited_categories[label] = {
                'palette': category_palette,
                'labels': labels,
                'indices': category_data['indices']
            }
            self.color_palette.extend(category_palette)

    def replace_excess_colors(self) -> None:
        new_pixels = np.zeros_like(self.pixels)

        # Iterate over each hue category
        for category in self.limited_categories.values():
            indices = category['indices']
            cluster_labels = category['labels']
            palette = category['palette']

            # Replace colors for pixels in this category
            for idx_in_category, original_idx in enumerate(indices):
                cluster_label = cluster_labels[idx_in_category]
                new_color = tuple(palette[cluster_label])

                # Only use colors that appear `min_color_usage` times or more
                if self.color_usage.get(new_color, 0) >= self.min_color_usage:
                    new_pixels[original_idx] = new_color
                else:
                    # Assign a default color (black or white) for colors used less than `min_color_usage` times
                    new_pixels[original_idx] = [0, 0, 0]  # Black as default for unused colors

        self.image_data = new_pixels.reshape(self.image_data.shape)

    def update_image(self) -> None:
        self.image = Image.fromarray(self.image_data.astype('uint8'), 'RGB')
        self.prepare_color_ranges()

    def prepare_color_ranges(self) -> None:
        self.color_ranges = [
            sorted(category['palette'], key=lambda rgb: self.rgb_to_hsv(rgb)[2], reverse=True)
            for category in self.limited_categories.values()
        ]

    @staticmethod
    def rgb_to_hsv(rgb: Union[np.ndarray, Tuple[int, int, int]]) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=float)
        r, g, b = rgb / 255.0
        maxc = max(r, g, b)
        minc = min(r, g, b)
        v = maxc

        if maxc == minc:
            return np.array([0.0, 0.0, v])

        s = (maxc - minc) / maxc
        rc = (maxc - r) / (maxc - minc)
        gc = (maxc - g) / (maxc - minc)
        bc = (maxc - b) / (maxc - minc)

        if r == maxc:
            h = bc - gc
        elif g == maxc:
            h = 2.0 + rc - bc
        else:
            h = 4.0 + gc - rc

        h = (h / 6.0) % 1.0
        return np.array([h * 360, s, v])

    @staticmethod
    def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
        h, s, v = hsv
        h = (h % 360) / 60.0

        f = h - int(h)
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))

        if int(h) == 0:
            r, g, b = v, t, p
        elif int(h) == 1:
            r, g, b = q, v, p
        elif int(h) == 2:
            r, g, b = p, v, t
        elif int(h) == 3:
            r, g, b = p, q, v
        elif int(h) == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q

        return np.array([int(x * 255) for x in (r, g, b)], dtype=int)
