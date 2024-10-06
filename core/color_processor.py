import numpy as np
from collections import Counter
from skimage.color import rgb2lab, deltaE_ciede2000
from sklearn.cluster import KMeans
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_float
from PIL import Image


class ColorProcessor:
    def __init__(self, image):
        # Convert image to RGB and get pixel data
        self.image = image.convert('RGB')
        self.image_data = np.array(self.image)
        self.pixels_rgb = self.image_data.reshape(-1, 3)

        # Precompute LAB and HSV values to avoid repeated conversions
        self.pixels_lab = rgb2lab(self.pixels_rgb.reshape(-1, 1, 3)).reshape(-1, 3)
        self.pixels_hsv = np.array([self.rgb_to_hsv(pixel) for pixel in self.pixels_rgb])

        self.total_pixels = len(self.pixels_rgb)
        self.color_palette = []
        self.color_ranges = []
        self.segments = None
        self.num_segments = 0

        # Configuration
        self.max_colors_per_category = 12
        self.max_categories = 10
        self.hue_tolerance = 15
        self.min_color_count = 20
        self.min_segment_size = 50

    def process_colors(self, progress_callback=None):
        steps = [
            (5, self.segment_image),
            (15, self.remove_infrequent_colors),
            (25, self.group_colors_by_hue),
            (40, self.limit_colors_within_categories),
            (100, self.process_segments)
        ]

        for progress, step in steps:
            if progress_callback:
                progress_callback.emit(progress)
            step()

    def segment_image(self):
        image_float = img_as_float(self.image)
        self.segments = felzenszwalb(image_float, scale=100, sigma=0.5, min_size=self.min_segment_size)
        self.num_segments = np.max(self.segments) + 1

    def remove_infrequent_colors(self):
        # Count occurrences of each color
        color_counts = Counter(map(tuple, self.pixels_rgb))
        self.frequent_colors = {color for color, count in color_counts.items() if count >= self.min_color_count}

        # Create a mask to filter out infrequent colors
        self.frequent_pixels_mask = np.array([tuple(pixel) in self.frequent_colors for pixel in self.pixels_rgb])

        # Filter only frequent colors
        self.pixels_frequent_rgb = self.pixels_rgb[self.frequent_pixels_mask]
        self.pixels_frequent_lab = self.pixels_lab[self.frequent_pixels_mask]
        self.pixels_frequent_hsv = self.pixels_hsv[self.frequent_pixels_mask]

    def group_colors_by_hue(self):
        # Group colors based on hue values in HSV space
        hues = self.pixels_frequent_hsv[:, 0]
        num_categories = min(len(np.unique(hues)), self.max_categories)

        if num_categories == 0:
            # Skip if no distinct hues are present
            self.hue_labels = np.zeros(len(hues), dtype=int)
            return

        kmeans = KMeans(n_clusters=num_categories, random_state=42)
        self.hue_labels = kmeans.fit_predict(hues.reshape(-1, 1))
        self.cluster_centers = kmeans.cluster_centers_

        # Create a dictionary to store hue categories
        self.hue_categories = {label: {'pixels': [], 'indices': []} for label in range(num_categories)}
        for idx, label in enumerate(self.hue_labels):
            self.hue_categories[label]['pixels'].append(self.pixels_frequent_rgb[idx])
            self.hue_categories[label]['indices'].append(idx)

    def limit_colors_within_categories(self):
        self.limited_categories = {}

        for label, category_data in self.hue_categories.items():
            hsv_values = np.array([self.rgb_to_hsv(pixel) for pixel in category_data['pixels']])
            num_colors = min(len(hsv_values), self.max_colors_per_category)

            if num_colors == 0:
                continue

            kmeans = KMeans(n_clusters=num_colors, random_state=42)
            labels = kmeans.fit_predict(hsv_values[:, 2].reshape(-1, 1))
            cluster_centers = kmeans.cluster_centers_

            category_palette = [
                self.hsv_to_rgb([self.cluster_centers[label][0], np.mean(hsv_values[:, 1]), center[0]])
                for center in cluster_centers
            ]

            self.limited_categories[label] = {
                'palette': category_palette,
                'labels': labels,
                'indices': category_data['indices']
            }
            self.color_palette.extend(category_palette)

        # Convert color palette to LAB for further processing
        self.color_palette_lab = rgb2lab(np.array(self.color_palette).reshape(-1, 1, 3)).reshape(-1, 3)

    def process_segments(self):
        new_pixels = np.zeros_like(self.pixels_rgb)

        for segment_label in range(self.num_segments):
            segment_mask = self.segments.reshape(-1) == segment_label
            segment_pixels = self.pixels_rgb[segment_mask]

            if len(segment_pixels) < self.min_segment_size:
                new_pixels[segment_mask] = segment_pixels
                continue

            segment_new_pixels = self.process_segment(segment_pixels)
            new_pixels[segment_mask] = segment_new_pixels

        self.image_data = new_pixels.reshape(self.image_data.shape)

    def process_segment(self, segment_pixels):
        segment_pixels_lab = rgb2lab(segment_pixels.reshape(-1, 1, 3)).reshape(-1, 3)
        new_segment_pixels = np.zeros_like(segment_pixels)

        for i, pixel_lab in enumerate(segment_pixels_lab):
            distances = [deltaE_ciede2000(pixel_lab, color_lab) for color_lab in self.color_palette_lab]
            nearest_color = self.color_palette[np.argmin(distances)]
            new_segment_pixels[i] = nearest_color

        return new_segment_pixels

    def update_image(self):
        self.image = Image.fromarray(self.image_data.astype('uint8'), 'RGB')
        self.prepare_color_ranges()

    def prepare_color_ranges(self):
        self.color_ranges = []
        for palette in self.limited_categories.values():
            palette_sorted = sorted(palette['palette'], key=lambda rgb: self.rgb_to_hsv(rgb)[2], reverse=True)
            self.color_ranges.append(palette_sorted)

    @staticmethod
    def rgb_to_hsv(rgb):
        rgb_normalized = rgb / 255.0
        maxc = rgb_normalized.max()
        minc = rgb_normalized.min()
        v = maxc
        delta = maxc - minc

        if delta == 0:
            return np.array([0.0, 0.0, v])

        s = delta / maxc
        rc = (maxc - rgb_normalized[0]) / delta
        gc = (maxc - rgb_normalized[1]) / delta
        bc = (maxc - rgb_normalized[2]) / delta

        if rgb_normalized[0] == maxc:
            h = bc - gc
        elif rgb_normalized[1] == maxc:
            h = 2.0 + rc - bc
        else:
            h = 4.0 + gc - rc

        h = (h / 6.0) % 1.0
        return np.array([h * 360, s, v])

    @staticmethod
    def hsv_to_rgb(hsv):
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

        return np.array([int(r * 255), int(g * 255), int(b * 255)])
