"""
Quick test script for single image prediction
Usage: python test_single_image.py path/to/image.jpg
"""

import sys
import cv2
import numpy as np
import json
from src.model import FruitCNN
from src.color_analyzer import ColorAnalyzer
from src.ripeness_detector import RipenessDetector

def load_model():
    """Load trained model"""
    with open('models/class_mappings.json', 'r') as f:
        mappings = json.load(f)
    
    num_classes = len(mappings['classes'])
    model = FruitCNN(input_shape=(100, 100, 3), num_classes=num_classes)
    model.load_model('models/fruit_model.h5')
    
    return model, mappings

def predict_image(image_path):
    """Predict fruit type and ripeness"""
    # Load model
    model, mappings = load_model()
    color_analyzer = ColorAnalyzer()
    ripeness_detector = RipenessDetector()
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Prepare for model
    img_resized = cv2.resize(img_rgb, (100, 100))
    img_normalized = img_resized.astype('float32') / 255.0
    
    # Get predictions
    predictions = model.predict(img_normalized)
    
    # Get top 3
    top_3_idx = np.argsort(predictions)[-3:][::-1]
    top_3_probs = predictions[top_3_idx]
    
    idx_to_class = {int(k): v for k, v in mappings['idx_to_class'].items()}
    top_3_classes = [idx_to_class[int(idx)] for idx in top_3_idx]
    
    # Color analysis
    color_info = color_analyzer.analyze_color_distribution(img_rgb)
    
    # Ripeness analysis
    ripeness_info = ripeness_detector.detect_ripeness(img_rgb, top_3_classes[0])
    
    # Print results
    print("\n" + "="*60)
    print("🍎 FRUIT RECOGNITION RESULTS")
    print("="*60)
    print(f"\n📷 Image: {image_path}")
    print(f"\n🎯 Predicted Fruit: {top_3_classes[0]}")
    print(f"   Confidence: {top_3_probs[0]*100:.2f}%")
    
    print(f"\n📊 Top 3 Predictions:")
    for i, (cls, prob) in enumerate(zip(top_3_classes, top_3_probs), 1):
        bar = "█" * int(prob * 30)
        print(f"   {i}. {cls:<30} [{bar:<30}] {prob*100:.2f}%")
    
    print(f"\n🎨 Color Analysis:")
    avg_rgb = color_info['average_rgb']
    print(f"   Average RGB: ({avg_rgb[0]}, {avg_rgb[1]}, {avg_rgb[2]})")
    print(f"   Color Name: {color_info['color_name'].title()}")
    
    print(f"\n🍌 Ripeness Analysis:")
    print(f"   Status: {ripeness_info['emoji']} {ripeness_info['level'].upper().replace('_', ' ')}")
    print(f"   Ripeness: {ripeness_info['percentage']}%")
    print(f"   Description: {ripeness_info['description']}")
    
    print("\n" + "="*60 + "\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_single_image.py path/to/image.jpg")
        print("\nExample:")
        print("  python test_single_image.py dataset/Test/Banana/1_100.jpg")
        return
    
    image_path = sys.argv[1]
    predict_image(image_path)

if __name__ == "__main__":
    main()