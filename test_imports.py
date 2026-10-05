"""
Test if all required packages are installed correctly
Run this before starting the project
"""

import sys

def test_import(package_name, import_statement):
    """Test if a package can be imported"""
    try:
        exec(import_statement)
        print(f"✅ {package_name:<20} - OK")
        return True
    except ImportError as e:
        print(f"❌ {package_name:<20} - FAILED")
        print(f"   Error: {e}")
        return False

def main():
    """Test all imports"""
    print("="*60)
    print("🔍 TESTING PACKAGE IMPORTS")
    print("="*60)
    print()
    
    tests = [
        ("TensorFlow", "import tensorflow as tf"),
        ("Keras", "from tensorflow import keras"),
        ("NumPy", "import numpy as np"),
        ("OpenCV", "import cv2"),
        ("Matplotlib", "import matplotlib.pyplot as plt"),
        ("Pillow", "from PIL import Image"),
        ("Scikit-learn", "from sklearn.model_selection import train_test_split"),
        ("Seaborn", "import seaborn as sns"),
        ("Pandas", "import pandas as pd"),
    ]
    
    passed = 0
    failed = 0
    
    for package, statement in tests:
        if test_import(package, statement):
            passed += 1
        else:
            failed += 1
        print()
    
    print("="*60)
    print(f"📊 RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n✅ ALL PACKAGES INSTALLED CORRECTLY!")
        print("🚀 You can now run the project!")
    else:
        print(f"\n❌ {failed} package(s) failed to import")
        print("\n📝 To fix, run:")
        print("   pip install -r requirements.txt")
    
    # Show Python version
    print(f"\n🐍 Python version: {sys.version}")
    
    # Show TensorFlow version if available
    try:
        import tensorflow as tf
        print(f"🤖 TensorFlow version: {tf.__version__}")
    except:
        print("⚠️  TensorFlow not installed")
    
    print()

if __name__ == "__main__":
    main()