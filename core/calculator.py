class Calculator:
    def __init__(self, density=45):
        self.density = density

    def calculate_size(self, width_px, height_px, density=None):
        if density is None:
            density = self.density
        width_cm = ((width_px / density) * 7) / 10
        height_cm = ((height_px / density) * 7) / 10
        return width_cm, height_cm

    def adjust_size(self, new_width_cm, aspect_ratio):
        new_height_cm = new_width_cm * aspect_ratio
        return new_width_cm, new_height_cm
