"""
Color analysis for fruits
"""

import cv2
import numpy as np
from collections import Counter

class ColorAnalyzer:
    def __init__(self):
        pass
    
    def get_dominant_color(self, image, k=3):
        """
        Get dominant color using K-means clustering
        """
        # Reshape image to list of pixels
        pixels = image.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Find most common color
        labels = labels.flatten()
        counter = Counter(labels)
        dominant = centers[counter.most_common(1)[0][0]]
        
        return dominant.astype(int)
    
    def get_average_color(self, image, mask=None):
        """
        Get average RGB color
        """
        if mask is not None:
            avg_color = cv2.mean(image, mask=mask)[:3]
        else:
            avg_color = cv2.mean(image)[:3]
        
        return np.array(avg_color, dtype=int)
    
    def rgb_to_hsv(self, rgb):
        """
        Convert RGB to HSV
        """
        rgb_normalized = rgb.reshape(1, 1, 3).astype('uint8')
        hsv = cv2.cvtColor(rgb_normalized, cv2.COLOR_RGB2HSV)
        return hsv[0][0]
    
    def get_color_name(self, rgb):
        """
        Get color name from RGB value
        Returns tuple: (name, english_name)
        """
        r, g, b = rgb
        
        # Define ranges for different colors
        if r > 200 and g < 100 and b < 100:
            return "red"
        elif r > 200 and g > 150 and b < 100:
            return "orange"
        elif r > 200 and g > 200 and b < 100:
            return "yellow"
        elif r < 100 and g > 150 and b < 100:
            return "green"
        elif r < 100 and g < 100 and b > 150:
            return "blue"
        elif r > 150 and g < 100 and b > 150:
            return "purple"
        elif r > 100 and g > 50 and b < 50:
            return "brown"
        elif r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        else:
            # Check for yellow-brown (overripe)
            if 100 < r < 200 and 80 < g < 150 and b < 100:
                return "yellow-brown"
            # Check for light green
            elif 100 < r < 180 and 150 < g < 220 and 50 < b < 150:
                return "light-green"
            return "mixed"
    
    def extract_fruit_mask(self, image):
        """
        Extract fruit mask (remove background)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Create mask for white background
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Invert mask (fruit = white, background = black)
        fruit_mask = cv2.bitwise_not(white_mask)
        
        # Morphological operations to clean mask
        kernel = np.ones((5, 5), np.uint8)
        fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_CLOSE, kernel)
        fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_OPEN, kernel)
        
        return fruit_mask
    
    def analyze_color_distribution(self, image):
        """
        Analyze color distribution
        """
        # Extract fruit without background
        mask = self.extract_fruit_mask(image)
        
        # Get dominant and average colors
        dominant_color = self.get_dominant_color(image)
        average_color = self.get_average_color(image, mask)
        
        # Get HSV values
        hsv = self.rgb_to_hsv(average_color)
        
        return {
            'dominant_rgb': dominant_color,
            'average_rgb': average_color,
            'hsv': hsv,
            'color_name': self.get_color_name(average_color)
        }
    
    def get_brightness(self, rgb):
        """
        Calculate brightness (0-255)
        """
        # Using perceived brightness formula
        r, g, b = rgb
        brightness = np.sqrt(0.299 * r**2 + 0.587 * g**2 + 0.114 * b**2)
        return brightness