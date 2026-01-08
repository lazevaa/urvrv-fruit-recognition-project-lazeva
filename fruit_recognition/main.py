"""
Main application for Fruit Recognition and Ripeness Detection
"""

import tkinter as tk
from tkinter import messagebox
import os
import json

from src.model import FruitCNN
from src.data_loader import FruitDataLoader
from src.color_analyzer import ColorAnalyzer
from src.ripeness_detector import RipenessDetector
from src.gui import FruitRecognitionGUI

def check_model_exists():
    """Check if trained model exists"""
    model_path = 'models/fruit_model.h5'
    mappings_path = 'models/class_mappings.json'
    
    if not os.path.exists(model_path):
        return False, "Model file not found"
    
    if not os.path.exists(mappings_path):
        return False, "Class mappings file not found"
    
    return True, None

def load_model_and_mappings():
    """Load trained model and class mappings"""
    print("🔄 Loading trained model...")
    
    # Load class mappings
    with open('models/class_mappings.json', 'r') as f:
        mappings = json.load(f)
    
    num_classes = len(mappings['classes'])
    
    # Initialize and load model
    model = FruitCNN(input_shape=(100, 100, 3), num_classes=num_classes)
    model.load_model('models/fruit_model.h5')
    
    # Create data loader with mappings
    data_loader = FruitDataLoader()
    data_loader.classes = mappings['classes']
    data_loader.class_to_idx = {k: int(v) for k, v in mappings['class_to_idx'].items()}
    data_loader.idx_to_class = {int(k): v for k, v in mappings['idx_to_class'].items()}
    
    print("✓ Model loaded successfully!")
    print(f"  - Number of classes: {num_classes}")
    
    return model, data_loader

def main():
    """Main application entry point"""
    print("="*60)
    print("🍎 FRUIT RECOGNITION & RIPENESS DETECTOR")
    print("="*60)
    print()
    
    # Check if model exists
    exists, error = check_model_exists()
    
    if not exists:
        print(f"❌ Error: {error}")
        print("\n📝 Please train the model first by running:")
        print("   python train_model.py")
        print("\n⚠️  Make sure you have:")
        print("   1. Downloaded the Fruit-360 dataset from Kaggle")
        print("   2. Extracted it to the 'dataset' folder")
        print("   3. The folder structure should be:")
        print("      dataset/")
        print("      ├── Training/")
        print("      │   ├── Apple Braeburn/")
        print("      │   ├── Banana/")
        print("      │   └── ...")
        print("      └── Test/")
        print("          ├── Apple Braeburn/")
        print("          ├── Banana/")
        print("          └── ...")
        return
    
    try:
        # Load model and mappings
        model, data_loader = load_model_and_mappings()
        
        # Initialize analyzers
        color_analyzer = ColorAnalyzer()
        ripeness_detector = RipenessDetector()
        
        print("\n🚀 Starting GUI application...")
        
        # Create GUI
        root = tk.Tk()
        app = FruitRecognitionGUI(
            root,
            model,
            data_loader,
            color_analyzer,
            ripeness_detector
        )
        
        print("✓ Application started successfully!")
        print("\n💡 Instructions:")
        print("   1. Click '📁 Load Image' to load a fruit image")
        print("   2. Click '🔍 Analyze' to identify the fruit and check ripeness")
        print("   3. Click '📂 Analyze Folder' to analyze multiple images")
        print("\n" + "="*60)
        
        # Run GUI
        root.mainloop()
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        messagebox.showerror("Error", f"Failed to start application:\n{str(e)}")

if __name__ == "__main__":
    main()