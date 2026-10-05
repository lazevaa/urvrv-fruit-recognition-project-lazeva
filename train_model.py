"""
Training script for Fruit Recognition Model
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from src.data_loader import FruitDataLoader
from src.model import FruitCNN

def plot_training_history(history, save_path='results/training_history.png'):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history plot saved to {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, save_path='results/confusion_matrix.png'):
    """Plot confusion matrix for top classes"""
    # Get top 20 most common classes
    unique, counts = np.unique(y_true, return_counts=True)
    top_20_idx = unique[np.argsort(counts)[-20:]]
    
    # Filter predictions for top 20 classes
    mask = np.isin(y_true, top_20_idx)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    # Create confusion matrix
    cm = confusion_matrix(y_true_filtered, y_pred_filtered)
    
    # Plot
    plt.figure(figsize=(15, 12))
    class_names = [classes[i] for i in top_20_idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Top 20 Classes)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()

def main():
    print("="*60)
    print("🍎 FRUIT RECOGNITION MODEL TRAINING")
    print("="*60)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Initialize data loader
    print("\n📊 Step 1: Loading data...")
    data_loader = FruitDataLoader(
        train_dir='dataset/Training',
        test_dir='dataset/Test',
        img_size=(100, 100)
    )
    
    # Prepare data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.prepare_data()
    num_classes = data_loader.get_num_classes()
    
    print(f"\n✓ Data loaded successfully!")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Validation samples: {len(X_val)}")
    print(f"  - Test samples: {len(X_test)}")
    
    # Build model
    print("\n🏗️  Step 2: Building CNN model...")
    model = FruitCNN(input_shape=(100, 100, 3), num_classes=num_classes)
    model.build_model()
    model.compile_model(learning_rate=0.001)
    
    print("\n📋 Model Architecture:")
    model.summary()
    
    # Train model
    print("\n🚀 Step 3: Training model...")
    print("This may take a while depending on your hardware...")
    
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32
    )
    
    # Plot training history
    print("\n📊 Step 4: Plotting training history...")
    plot_training_history(history)
    
    # Evaluate on test set
    print("\n🎯 Step 5: Evaluating on test set...")
    results = model.evaluate(X_test, y_test)
    
    # Get predictions for confusion matrix
    print("\n📊 Step 6: Generating confusion matrix...")
    y_pred_probs = model.model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, data_loader.classes)
    
    # Save classification report
    print("\n📄 Step 7: Generating classification report...")
    report = classification_report(
        y_true, y_pred,
        target_names=data_loader.classes,
        digits=3
    )
    
    with open('results/classification_report.txt', 'w') as f:
        f.write("FRUIT RECOGNITION - CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    
    print("✓ Classification report saved to results/classification_report.txt")
    
    # Save model
    print("\n💾 Step 8: Saving model...")
    model.save_model('models/fruit_model.h5')
    
    # Save class mappings
    import json
    with open('models/class_mappings.json', 'w') as f:
        json.dump({
            'classes': data_loader.classes,
            'class_to_idx': data_loader.class_to_idx,
            'idx_to_class': data_loader.idx_to_class
        }, f, indent=2)
    
    print("✓ Class mappings saved to models/class_mappings.json")
    
    # Final summary
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\n📊 Final Results:")
    print(f"  - Test Accuracy: {results[1]*100:.2f}%")
    print(f"  - Test Top-3 Accuracy: {results[2]*100:.2f}%")
    print(f"\n📁 Saved Files:")
    print(f"  - Model: models/fruit_model.h5")
    print(f"  - Class mappings: models/class_mappings.json")
    print(f"  - Training history plot: results/training_history.png")
    print(f"  - Confusion matrix: results/confusion_matrix.png")
    print(f"  - Classification report: results/classification_report.txt")
    print("\n🎉 You can now run main.py to use the trained model!")
    print("="*60)

if __name__ == "__main__":
    main()