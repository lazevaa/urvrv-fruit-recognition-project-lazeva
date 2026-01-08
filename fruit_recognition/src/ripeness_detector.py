"""
Ripeness detection based on color analysis
"""

import numpy as np
from src.color_analyzer import ColorAnalyzer

class RipenessDetector:
    def __init__(self):
        self.color_analyzer = ColorAnalyzer()
        
        # Define ripeness rules for different fruits
        self.ripeness_rules = {
            'Banana': self.analyze_banana,
            'Apple': self.analyze_apple,
            'Avocado': self.analyze_avocado,
            'Tomato': self.analyze_tomato,
            'Orange': self.analyze_orange,
            'Lemon': self.analyze_lemon,
            'Mango': self.analyze_mango,
            'Pear': self.analyze_pear,
            'Strawberry': self.analyze_strawberry,
            'Peach': self.analyze_peach,
            'Kiwi': self.analyze_kiwi,
            'Plum': self.analyze_plum,
        }
    
    def detect_ripeness(self, image, fruit_name):
        """
        Main function to detect ripeness
        Returns: dict with ripeness level and description
        """
        # Get color analysis
        color_info = self.color_analyzer.analyze_color_distribution(image)
        avg_rgb = color_info['average_rgb']
        color_name = color_info['color_name']
        hsv = color_info['hsv']
        
        # Find matching rule (partial match)
        for key in self.ripeness_rules.keys():
            if key.lower() in fruit_name.lower():
                return self.ripeness_rules[key](avg_rgb, color_name, hsv)
        
        # Default analysis if no specific rule
        return self.analyze_generic(avg_rgb, color_name, hsv)
    
    def analyze_banana(self, rgb, color_name, hsv):
        """Analyze banana ripeness"""
        r, g, b = rgb
        h, s, v = hsv
        
        # Green banana (unripe)
        if g > r and g > b and s > 100:
            return {
                'level': 'unripe',
                'percentage': 20,
                'description': 'Green banana - not ready to eat. Wait a few days.',
                'emoji': '🟢'
            }
        
        # Yellow banana (ripe)
        elif r > 180 and g > 180 and b < 120 and 20 < h < 40:
            return {
                'level': 'ripe',
                'percentage': 85,
                'description': 'Yellow banana - perfect for eating! Sweet and delicious.',
                'emoji': '🟡'
            }
        
        # Brown spots or dark yellow (overripe)
        elif (r > 150 and g > 100 and b < 100) or (r < 150 and g < 100):
            return {
                'level': 'overripe',
                'percentage': 95,
                'description': 'Brown-spotted banana - overripe. Good for baking or smoothies.',
                'emoji': '🟤'
            }
        
        # Light yellow (slightly unripe)
        else:
            return {
                'level': 'slightly_unripe',
                'percentage': 60,
                'description': 'Light yellow banana - almost ready. Wait 1-2 days.',
                'emoji': '🟨'
            }
    
    def analyze_apple(self, rgb, color_name, hsv):
        """Analyze apple ripeness"""
        r, g, b = rgb
        
        # Green apple (unripe or green variety)
        if g > r and g > b:
            return {
                'level': 'ripe',
                'percentage': 80,
                'description': 'Green apple - crisp and tart. Ready to eat!',
                'emoji': '🍏'
            }
        
        # Red apple (ripe)
        elif r > 150 and r > g:
            return {
                'level': 'ripe',
                'percentage': 90,
                'description': 'Red apple - sweet and juicy. Perfect for eating!',
                'emoji': '🍎'
            }
        
        # Brownish (overripe)
        elif r > 100 and g > 80 and b < 80:
            return {
                'level': 'overripe',
                'percentage': 95,
                'description': 'Brownish apple - overripe. Check for soft spots.',
                'emoji': '🟤'
            }
        
        else:
            return {
                'level': 'ripe',
                'percentage': 85,
                'description': 'Apple looks good - ready to eat!',
                'emoji': '🍎'
            }
    
    def analyze_avocado(self, rgb, color_name, hsv):
        """Analyze avocado ripeness"""
        r, g, b = rgb
        brightness = self.color_analyzer.get_brightness(rgb)
        
        # Bright green (unripe)
        if g > 150 and brightness > 100:
            return {
                'level': 'unripe',
                'percentage': 30,
                'description': 'Bright green avocado - hard and unripe. Wait 3-5 days.',
                'emoji': '🟢'
            }
        
        # Dark green (ripe)
        elif 50 < g < 120 and brightness < 100:
            return {
                'level': 'ripe',
                'percentage': 85,
                'description': 'Dark green avocado - ready to eat! Should yield to gentle pressure.',
                'emoji': '🥑'
            }
        
        # Very dark or black (overripe)
        elif brightness < 50:
            return {
                'level': 'overripe',
                'percentage': 95,
                'description': 'Very dark avocado - may be overripe. Check inside for brown spots.',
                'emoji': '⚫'
            }
        
        else:
            return {
                'level': 'slightly_unripe',
                'percentage': 65,
                'description': 'Avocado is getting ready - wait 1-2 days.',
                'emoji': '🟢'
            }
    
    def analyze_tomato(self, rgb, color_name, hsv):
        """Analyze tomato ripeness"""
        r, g, b = rgb
        
        # Green (unripe)
        if g > r and g > 120:
            return {
                'level': 'unripe',
                'percentage': 25,
                'description': 'Green tomato - unripe. Not ready for eating raw.',
                'emoji': '🟢'
            }
        
        # Red (ripe)
        elif r > 180 and r > g * 1.5:
            return {
                'level': 'ripe',
                'percentage': 90,
                'description': 'Red tomato - perfectly ripe! Great for salads.',
                'emoji': '🍅'
            }
        
        # Orange-red (ripening)
        elif r > 150 and g > 80 and r > g:
            return {
                'level': 'slightly_unripe',
                'percentage': 70,
                'description': 'Orange-red tomato - almost ripe. Wait 1 day.',
                'emoji': '🟠'
            }
        
        # Dark red (overripe)
        elif r > 150 and r < 200 and g < 80:
            return {
                'level': 'overripe',
                'percentage': 95,
                'description': 'Dark red tomato - very ripe. Use soon or make sauce.',
                'emoji': '🔴'
            }
        
        else:
            return {
                'level': 'ripe',
                'percentage': 80,
                'description': 'Tomato looks good!',
                'emoji': '🍅'
            }
    
    def analyze_orange(self, rgb, color_name, hsv):
        """Analyze orange ripeness"""
        r, g, b = rgb
        
        # Green tint (unripe)
        if g > 120 and g > r * 0.8:
            return {
                'level': 'unripe',
                'percentage': 40,
                'description': 'Greenish orange - not fully ripe. Might be sour.',
                'emoji': '🟢'
            }
        
        # Bright orange (ripe)
        elif r > 200 and 120 < g < 180 and b < 100:
            return {
                'level': 'ripe',
                'percentage': 90,
                'description': 'Bright orange - perfectly ripe and juicy!',
                'emoji': '🍊'
            }
        
        # Dark orange (very ripe)
        elif r > 180 and g < 120:
            return {
                'level': 'overripe',
                'percentage': 85,
                'description': 'Dark orange - very ripe. Eat soon!',
                'emoji': '🟠'
            }
        
        else:
            return {
                'level': 'ripe',
                'percentage': 85,
                'description': 'Orange looks good - ready to eat!',
                'emoji': '🍊'
            }
    
    def analyze_lemon(self, rgb, color_name, hsv):
        """Analyze lemon ripeness"""
        r, g, b = rgb
        
        # Green (unripe)
        if g > r and g > 150:
            return {
                'level': 'unripe',
                'percentage': 40,
                'description': 'Green lemon - unripe. Very sour and less juicy.',
                'emoji': '🟢'
            }
        
        # Yellow (ripe)
        elif r > 200 and g > 200 and b < 120:
            return {
                'level': 'ripe',
                'percentage': 90,
                'description': 'Yellow lemon - perfectly ripe! Full of juice.',
                'emoji': '🍋'
            }
        
        # Dark yellow (very ripe)
        elif r > 180 and g > 150 and b < 100:
            return {
                'level': 'ripe',
                'percentage': 95,
                'description': 'Deep yellow lemon - very ripe. Use soon!',
                'emoji': '🟡'
            }
        
        else:
            return {
                'level': 'ripe',
                'percentage': 85,
                'description': 'Lemon looks good!',
                'emoji': '🍋'
            }
    
    def analyze_mango(self, rgb, color_name, hsv):
        """Analyze mango ripeness"""
        r, g, b = rgb
        
        # Green (unripe)
        if g > 150 and g > r:
            return {
                'level': 'unripe',
                'percentage': 30,
                'description': 'Green mango - hard and unripe. Wait several days.',
                'emoji': '🟢'
            }
        
        # Orange-red (ripe)
        elif r > 150 and r > g:
            return {
                'level': 'ripe',
                'percentage': 90,
                'description': 'Orange-red mango - perfectly ripe! Sweet and soft.',
                'emoji': '🥭'
            }
        
        # Mixed yellow-green (ripening)
        elif abs(r - g) < 50 and r > 120:
            return {
                'level': 'slightly_unripe',
                'percentage': 65,
                'description': 'Yellowing mango - almost ripe. Wait 1-2 days.',
                'emoji': '🟡'
            }
        
        else:
            return {
                'level': 'ripe',
                'percentage': 80,
                'description': 'Mango looks good!',
                'emoji': '🥭'
            }
    
    def analyze_pear(self, rgb, color_name, hsv):
        """Analyze pear ripeness"""
        r, g, b = rgb
        
        # Green (unripe or green variety)
        if g > 150 and g > r:
            return {
                'level': 'slightly_unripe',
                'percentage': 70,
                'description': 'Green pear - may be unripe or a green variety. Check for softness.',
                'emoji': '🍐'
            }
        
        # Yellow-green (ripe)
        elif r > 150 and g > 150 and abs(r - g) < 50:
            return {
                'level': 'ripe',
                'percentage': 90,
                'description': 'Yellow-green pear - perfectly ripe! Should be slightly soft.',
                'emoji': '🍐'
            }
        
        # Brown spots (overripe)
        elif r > 100 and g > 80 and b < 80:
            return {
                'level': 'overripe',
                'percentage': 95,
                'description': 'Brownish pear - overripe. May have soft spots.',
                'emoji': '🟤'
            }
        
        else:
            return {
                'level': 'ripe',
                'percentage': 85,
                'description': 'Pear looks good!',
                'emoji': '🍐'
            }
    
    def analyze_strawberry(self, rgb, color_name, hsv):
        """Analyze strawberry ripeness"""
        r, g, b = rgb
        
        # White or light pink (unripe)
        if r < 150 and g > 100:
            return {
                'level': 'unripe',
                'percentage': 30,
                'description': 'Light-colored strawberry - unripe. Will be sour.',
                'emoji': '🤍'
            }
        
        # Bright red (ripe)
        elif r > 180 and r > g * 2:
            return {
                'level': 'ripe',
                'percentage': 90,
                'description': 'Bright red strawberry - perfectly ripe! Sweet and juicy.',
                'emoji': '🍓'
            }
        
        # Dark red (very ripe)
        elif r > 150 and r < 200 and g < 80:
            return {
                'level': 'overripe',
                'percentage': 95,
                'description': 'Dark red strawberry - very ripe. Eat immediately!',
                'emoji': '🔴'
            }
        
        else:
            return {
                'level': 'ripe',
                'percentage': 85,
                'description': 'Strawberry looks good!',
                'emoji': '🍓'
            }
    
    def analyze_peach(self, rgb, color_name, hsv):
        """Analyze peach ripeness"""
        r, g, b = rgb
        
        # Green tint (unripe)
        if g > 120 and g > r * 0.7:
            return {
                'level': 'unripe',
                'percentage': 40,
                'description': 'Greenish peach - unripe and hard. Wait a few days.',
                'emoji': '🟢'
            }
        
        # Orange-pink (ripe)
        elif r > 180 and g > 100 and r > g:
            return {
                'level': 'ripe',
                'percentage': 90,
                'description': 'Orange-pink peach - perfectly ripe! Soft and sweet.',
                'emoji': '🍑'
            }
        
        # Very soft texture implied by darker color
        elif r > 150 and g < 100:
            return {
                'level': 'overripe',
                'percentage': 95,
                'description': 'Dark peach - may be overripe. Check for mushiness.',
                'emoji': '🟤'
            }
        
        else:
            return {
                'level': 'ripe',
                'percentage': 85,
                'description': 'Peach looks good!',
                'emoji': '🍑'
            }
    
    def analyze_kiwi(self, rgb, color_name, hsv):
        """Analyze kiwi ripeness"""
        r, g, b = rgb
        brightness = self.color_analyzer.get_brightness(rgb)
        
        # Bright brown (unripe)
        if brightness > 120:
            return {
                'level': 'unripe',
                'percentage': 40,
                'description': 'Firm kiwi - unripe. Should be hard to the touch.',
                'emoji': '🟫'
            }
        
        # Medium brown (ripe)
        elif 70 < brightness < 120:
            return {
                'level': 'ripe',
                'percentage': 90,
                'description': 'Kiwi is ripe - should yield to gentle pressure. Perfect!',
                'emoji': '🥝'
            }
        
        # Dark or soft (overripe)
        elif brightness < 70:
            return {
                'level': 'overripe',
                'percentage': 95,
                'description': 'Soft kiwi - may be overripe. Check for mushiness.',
                'emoji': '🟤'
            }
        
        else:
            return {
                'level': 'ripe',
                'percentage': 85,
                'description': 'Kiwi looks good!',
                'emoji': '🥝'
            }
    
    def analyze_plum(self, rgb, color_name, hsv):
        """Analyze plum ripeness"""
        r, g, b = rgb
        
        # Light purple or greenish (unripe)
        if (r < 120 and b < 120) or (g > 120):
            return {
                'level': 'unripe',
                'percentage': 40,
                'description': 'Light-colored plum - unripe and firm. Wait a few days.',
                'emoji': '🟣'
            }
        
        # Deep purple/red (ripe)
        elif (r > 100 and b > 80) or r > 150:
            return {
                'level': 'ripe',
                'percentage': 90,
                'description': 'Deep-colored plum - ripe and juicy! Perfect for eating.',
                'emoji': '🍇'
            }
        
        # Very dark (overripe)
        elif r < 80 and g < 80 and b < 80:
            return {
                'level': 'overripe',
                'percentage': 95,
                'description': 'Very dark plum - may be overripe. Check for soft spots.',
                'emoji': '⚫'
            }
        
        else:
            return {
                'level': 'ripe',
                'percentage': 85,
                'description': 'Plum looks good!',
                'emoji': '🍇'
            }
    
    def analyze_generic(self, rgb, color_name, hsv):
        """Generic ripeness analysis for fruits without specific rules"""
        brightness = self.color_analyzer.get_brightness(rgb)
        
        if 'green' in color_name:
            return {
                'level': 'possibly_unripe',
                'percentage': 50,
                'description': 'Green color detected - may be unripe or a green variety.',
                'emoji': '🟢'
            }
        elif 'brown' in color_name or brightness < 60:
            return {
                'level': 'possibly_overripe',
                'percentage': 90,
                'description': 'Dark color detected - may be overripe.',
                'emoji': '🟤'
            }
        else:
            return {
                'level': 'ripe',
                'percentage': 80,
                'description': 'Fruit appears to be ripe based on color.',
                'emoji': '🍎'
            }