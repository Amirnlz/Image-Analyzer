import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_float


class ColorProcessor:
    def __init__(self, image):
        self.image = image
        self.color_ranges = []
        self.max_colors_per_category = 12
        self.max_categories = 10
        self.hue_tolerance = 15  # Degrees of hue tolerance for grouping
        self.min_color_count = 20  # Minimum number of occurrences for a color to be considered

    def process_colors(self, progress_callback=None):
        if progress_callback:
            progress_callback.emit(10)
        self.extract_colors()

        if progress_callback:
            progress_callback.emit(30)
        self.remove_infrequent_colors()

        if progress_callback:
            progress_callback.emit(50)
        self.group_colors_by_hue()

        if progress_callback:
            progress_callback.emit(70)
        self.limit_colors_within_categories()

        if progress_callback:
            progress_callback.emit(90)
        self.replace_excess_colors()

        if progress_callback:
            progress_callback.emit(100)

    def extract_colors(self):
        self.image = self.image.convert('RGB')
        self.image_data = np.array(self.image)
        self.pixels = self.image_data.reshape(-1, 3)
        self.total_pixels = len(self.pixels)

    def remove_infrequent_colors(self):
        # Count occurrences of each color
        pixel_list = [tuple(pixel) for pixel in self.pixels]
        color_counts = Counter(pixel_list)

        # Identify colors that appear less than min_color_count times
        self.frequent_colors = set(color for color, count in color_counts.items() if count >= self.min_color_count)
        self.infrequent_colors = set(color_counts.keys()) - self.frequent_colors

        # Create a mask for frequent colors
        self.frequent_pixels_mask = np.array([tuple(pixel) in self.frequent_colors for pixel in self.pixels])

        # For infrequent colors, we'll process them in replace_excess_colors
        # For now, we proceed with frequent pixels
        self.pixels_frequent = self.pixels[self.frequent_pixels_mask]

    def group_colors_by_hue(self):
        # Convert frequent pixels to HSV
        self.pixels_frequent_hsv = np.array([self.rgb_to_hsv(pixel) for pixel in self.pixels_frequent])

        # Group colors based on hue
        hues = self.pixels_frequent_hsv[:, 0]
        num_categories = min(len(np.unique(hues)), self.max_categories)
        kmeans = KMeans(n_clusters=num_categories, random_state=42)
        kmeans.fit(hues.reshape(-1, 1))
        self.hue_labels = kmeans.labels_
        self.cluster_centers = kmeans.cluster_centers_

        # Create hue categories and store pixel indices
        self.hue_categories = {}
        for idx, label in enumerate(self.hue_labels):
            pixel_rgb = self.pixels_frequent[idx]
            pixel_hsv = self.pixels_frequent_hsv[idx]
            if label not in self.hue_categories:
                self.hue_categories[label] = {'pixels': [], 'indices': []}
            self.hue_categories[label]['pixels'].append({'rgb': pixel_rgb, 'hsv': pixel_hsv})
            self.hue_categories[label]['indices'].append(idx)

    def limit_colors_within_categories(self):
        self.limited_categories = {}
        self.color_palette = []
        for label, category_data in self.hue_categories.items():
            hsv_values = np.array([pixel['hsv'] for pixel in category_data['pixels']])
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
                hsv = np.array([self.cluster_centers[label][0], np.mean(hsv_values[:, 1]), center[0]])
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

        # Start with frequent pixels
        frequent_indices = np.where(self.frequent_pixels_mask)[0]
        for label, category in self.limited_categories.items():
            indices_in_category = [frequent_indices[idx] for idx in category['indices']]
            cluster_labels = category['labels']
            palette = category['palette']

            # Replace colors for pixels in this category
            for idx_in_category, original_idx in enumerate(indices_in_category):
                cluster_label = cluster_labels[idx_in_category]
                new_color = palette[cluster_label]
                new_pixels[original_idx] = new_color

        # Handle infrequent colors
        infrequent_indices = np.where(~self.frequent_pixels_mask)[0]
        infrequent_pixels = self.pixels[~self.frequent_pixels_mask]

        # Map infrequent colors to the nearest frequent color
        for idx, pixel in zip(infrequent_indices, infrequent_pixels):
            # Convert to HSV
            pixel_hsv = self.rgb_to_hsv(pixel)
            # Find the nearest color in the palette
            distances = [self.color_distance(pixel_hsv, self.rgb_to_hsv(color)) for color in self.color_palette]
            nearest_color = self.color_palette[np.argmin(distances)]
            new_pixels[idx] = nearest_color

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
            palette_sorted = sorted(palette, key=lambda rgb: self.rgb_to_hsv(rgb)[2], reverse=True)
            self.color_ranges.append(palette_sorted)

    @staticmethod
    def rgb_to_hsv(rgb):
        # Convert RGB to HSV
        r, g, b = rgb / 255.0
        maxc = max(r, g, b)
        minc = min(r, g, b)
        v = maxc
        delta = maxc - minc
        if delta == 0:
            h = 0.0
            s = 0.0
        else:
            s = delta / maxc
            if r == maxc:
                h = (g - b) / delta % 6
            elif g == maxc:
                h = (b - r) / delta + 2
            else:
                h = (r - g) / delta + 4
            h *= 60
        return np.array([h, s, v])

    @staticmethod
    def hsv_to_rgb(hsv):
        # Convert HSV to RGB
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

    @staticmethod
    def color_distance(hsv1, hsv2):
        # Calculate the Euclidean distance between two HSV colors
        return np.linalg.norm(hsv1 - hsv2)
