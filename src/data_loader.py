"""
Data Loader for Fruit-360 dataset
Loading and preparing data
"""

import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class FruitDataLoader:
    def __init__(self, train_dir='dataset/Training', test_dir='dataset/Test', img_size=(100, 100)):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.img_size = img_size
        self.classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
    def load_classes(self):
        """Load all classes (fruit types)"""
        if os.path.exists(self.train_dir):
            self.classes = sorted([d for d in os.listdir(self.train_dir) 
                                 if os.path.isdir(os.path.join(self.train_dir, d))])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
            print(f"✓ Loaded {len(self.classes)} fruit classes")
        else:
            raise FileNotFoundError(f"Cannot find: {self.train_dir}")
        
    def load_images_from_folder(self, folder):
        """Load images from folder"""
        images = []
        labels = []
        
        for class_name in self.classes:
            class_path = os.path.join(folder, class_name)
            if not os.path.exists(class_path):
                continue
                
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_name)
                    
                    # Load image
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, self.img_size)
                        images.append(img)
                        labels.append(class_idx)
        
        return np.array(images), np.array(labels)
    
    def prepare_data(self, validation_split=0.2):
        """Prepare data for training"""
        print("📊 Loading data...")
        
        # Load classes
        self.load_classes()
        
        # Load training data
        X_train, y_train = self.load_images_from_folder(self.train_dir)
        print(f"✓ Training: {len(X_train)} images")
        
        # Load test data
        X_test, y_test = self.load_images_from_folder(self.test_dir)
        print(f"✓ Test: {len(X_test)} images")
        
        # Normalize pixels (0-1)
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Split training into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_split, random_state=42
        )
        
        # One-hot encoding for labels
        num_classes = len(self.classes)
        y_train = to_categorical(y_train, num_classes)
        y_val = to_categorical(y_val, num_classes)
        y_test = to_categorical(y_test, num_classes)
        
        print(f"✓ Data preparation completed!")
        print(f"  - Training set: {X_train.shape}")
        print(f"  - Validation set: {X_val.shape}")
        print(f"  - Test set: {X_test.shape}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def get_class_name(self, class_idx):
        """Get class name from index"""
        return self.idx_to_class.get(class_idx, "Unknown")
    
    def get_num_classes(self):
        """Number of classes"""
        return len(self.classes)