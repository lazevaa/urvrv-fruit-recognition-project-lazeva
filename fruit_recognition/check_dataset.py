"""
Dataset Structure Checker
Verifies that Fruit-360 dataset is properly installed
"""

import os
from pathlib import Path
from collections import Counter

def check_dataset_structure():
    """Check if dataset has correct structure"""
    print("="*60)
    print("🔍 CHECKING DATASET STRUCTURE")
    print("="*60)
    
    errors = []
    warnings = []
    
    # Check if dataset folder exists
    if not os.path.exists('dataset'):
        errors.append("❌ 'dataset' folder not found!")
        errors.append("   Please create it and extract Fruit-360 dataset there.")
        return errors, warnings
    
    print("\n✓ Found 'dataset' folder")
    
    # Check Training folder
    train_dir = 'dataset/Training'
    if not os.path.exists(train_dir):
        errors.append(f"❌ '{train_dir}' folder not found!")
    else:
        print(f"✓ Found '{train_dir}' folder")
        
        # Count subfolders (fruit types)
        train_classes = [d for d in os.listdir(train_dir) 
                        if os.path.isdir(os.path.join(train_dir, d))]
        
        if len(train_classes) == 0:
            errors.append(f"❌ No fruit folders found in '{train_dir}'")
        else:
            print(f"✓ Found {len(train_classes)} fruit types in Training")
            
            # Count images
            total_train_images = 0
            for fruit in train_classes[:5]:  # Sample first 5
                fruit_path = os.path.join(train_dir, fruit)
                images = [f for f in os.listdir(fruit_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                total_train_images += len(images)
            
            print(f"✓ Sample check: ~{total_train_images} images in first 5 folders")
    
    # Check Test folder
    test_dir = 'dataset/Test'
    if not os.path.exists(test_dir):
        warnings.append(f"⚠️  '{test_dir}' folder not found (optional but recommended)")
    else:
        print(f"✓ Found '{test_dir}' folder")
        
        test_classes = [d for d in os.listdir(test_dir) 
                       if os.path.isdir(os.path.join(test_dir, d))]
        
        if len(test_classes) == 0:
            warnings.append(f"⚠️  No fruit folders found in '{test_dir}'")
        else:
            print(f"✓ Found {len(test_classes)} fruit types in Test")
    
    # Expected structure info
    print(f"\n📋 Expected Structure (Mini Dataset - 7 fruits):")
    print(f"   dataset/")
    print(f"   ├── Training/")
    print(f"   │   ├── Apple/")
    print(f"   │   ├── Banana/")
    print(f"   │   ├── Cherry/")
    print(f"   │   ├── Kiwi/")
    print(f"   │   ├── Lemon/")
    print(f"   │   ├── Mango/")
    print(f"   │   └── Pear/")
    print(f"   └── Test/")
    print(f"       ├── Apple/")
    print(f"       ├── Banana/")
    print(f"       └── ... (same 7 fruits)")
    
    return errors, warnings

def show_sample_fruits():
    """Show sample of available fruit types"""
    train_dir = 'dataset/Training'
    if not os.path.exists(train_dir):
        return
    
    classes = sorted([d for d in os.listdir(train_dir) 
                     if os.path.isdir(os.path.join(train_dir, d))])
    
    if len(classes) > 0:
        print(f"\n🍎 Sample Fruit Types (showing first 20 of {len(classes)}):")
        for i, fruit in enumerate(classes[:20], 1):
            fruit_path = os.path.join(train_dir, fruit)
            num_images = len([f for f in os.listdir(fruit_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"   {i:2d}. {fruit:<30} ({num_images} images)")
        
        if len(classes) > 20:
            print(f"   ... and {len(classes) - 20} more fruit types")

def main():
    """Main checker function"""
    errors, warnings = check_dataset_structure()
    
    # Show results
    if errors:
        print(f"\n❌ ERRORS FOUND:")
        for error in errors:
            print(f"   {error}")
        
        print(f"\n📥 How to fix:")
        print(f"   1. Download Fruit-360 dataset from:")
        print(f"      https://www.kaggle.com/datasets/moltean/fruits")
        print(f"   2. Extract the downloaded archive")
        print(f"   3. Rename the extracted folder to 'dataset'")
        print(f"   4. Make sure it contains 'Training' and 'Test' folders")
        print(f"   5. Run this checker again")
    
    if warnings:
        print(f"\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"   {warning}")
    
    if not errors and not warnings:
        print(f"\n✅ DATASET IS PROPERLY CONFIGURED!")
        print(f"   You can now run: python train_model.py")
        show_sample_fruits()
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()