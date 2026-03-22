import numpy as np
import cv2
import math
import random

class UnitCircleGenerator:
    def __init__(self, img_size=64, line_thickness=1, noise_level=0.0):
        """
        Args:
            img_size (int): Size of the square image (img_size x img_size).
            line_thickness (int): Base thickness for lines.
            noise_level (float): Probability/Intensity of noise artifacts.
        """
        self.img_size = img_size
        self.center = (img_size // 2, img_size // 2)
        self.radius = int(img_size * 0.4) # Radius is 40% of image size
        self.line_thickness = line_thickness
        self.noise_level = noise_level

    def generate_sample(self):
        """
        Generates a single synthetic unit circle image and its labels.
        
        Returns:
            image (np.array): Grayscale image of shape (img_size, img_size, 1), normalized to [0, 1].
            label (dict): Dictionary containing 'theta', 'sin', 'cos', 'quadrant'.
        """
        # Initialize blank white image (or random light gray)
        # 255 is white.
        bg_color = random.randint(240, 255)
        image = np.ones((self.img_size, self.img_size), dtype=np.uint8) * bg_color
        
        # Random parameters for this sample
        theta = random.uniform(0, 2 * math.pi)
        
        # Calculate target point
        cos_val = math.cos(theta)
        sin_val = math.sin(theta)
        
        target_x = int(self.center[0] + self.radius * cos_val)
        target_y = int(self.center[1] - self.radius * sin_val) # y-axis is down in image coords
        
        # --- Drawing ---
        
        # 1. Draw Axes (Optional noise: sometimes faint or missing?)
        # For now, always draw them but maybe vary intensity
        axis_color = random.randint(0, 100)
        thick = max(1, self.line_thickness + random.randint(-1, 1))
        
        # X-axis
        cv2.line(image, (0, self.center[1]), (self.img_size, self.center[1]), axis_color, 1)
        # Y-axis
        cv2.line(image, (self.center[0], 0), (self.center[0], self.img_size), axis_color, 1)
        
        # 2. Draw Circle
        color_circle = random.randint(0, 50)
        cv2.circle(image, self.center, self.radius, color_circle, 1)
        
        # 3. Draw Radius (The Angle Indicator)
        # Line from center to target
        color_line = random.randint(0, 20) # Darker
        cv2.line(image, self.center, (target_x, target_y), color_line, self.line_thickness)
        
        # 4. Draw Metric Marker (Dot at the end)
        cv2.circle(image, (target_x, target_y), 2, color_line, -1)
        
        # --- Noise Injection ---
        if self.noise_level > 0:
            # Salt and pepper noise
            noise = np.random.normal(0, 10, image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Normalize to 0-1 float
        image_norm = image.astype(np.float32) / 255.0
        image_norm = np.expand_dims(image_norm, axis=-1) # (H, W, 1)

        # Determine Quadrant (1, 2, 3, 4)
        # Standard math quadrants: Q1 (+x, +y), Q2 (-x, +y), Q3 (-x, -y), Q4 (+x, -y)
        # Note: sin is positive in Q1, Q2. cos is positive in Q1, Q4.
        if cos_val >= 0 and sin_val >= 0:
            quadrant = 0 # Q1
        elif cos_val < 0 and sin_val >= 0:
            quadrant = 1 # Q2
        elif cos_val < 0 and sin_val < 0:
            quadrant = 2 # Q3
        else:
            quadrant = 3 # Q4

        label = {
            'theta': theta,
            'sin': sin_val,
            'cos': cos_val,
            'quadrant': quadrant
        }
        
        return image_norm, label

if __name__ == "__main__":
    # Test block
    print("Testing generator...")
    gen = UnitCircleGenerator(img_size=128, noise_level=0.1)
    img, lbl = gen.generate_sample()
    print(f"Generated sample with label: {lbl}")
    print(f"Image shape: {img.shape}, range: [{img.min():.2f}, {img.max():.2f}]")
    
    # Save a debug image
    debug_img = (img * 255).astype(np.uint8)
    cv2.imwrite("debug_sample.png", debug_img)
    print("Saved debug_sample.png")
