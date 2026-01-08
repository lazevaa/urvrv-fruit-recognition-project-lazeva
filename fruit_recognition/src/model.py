"""
CNN Model for fruit classification
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

class FruitCNN:
    def __init__(self, input_shape=(100, 100, 3), num_classes=131):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        """Build CNN architecture"""
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        print("✓ CNN model built successfully!")
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
        )
        
        print("✓ Model compiled successfully!")
        
    def get_callbacks(self, model_path='models/fruit_model.h5'):
        """Callbacks for training"""
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint
            ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model"""
        print("🚀 Starting training...")
        
        # Data augmentation
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        
        # Augmented training data
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(1024).batch(batch_size)
        train_dataset = train_dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Validation dataset
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        print("✓ Training completed!")
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        results = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\n📊 Test set results:")
        print(f"  - Accuracy: {results[1]*100:.2f}%")
        print(f"  - Top-3 Accuracy: {results[2]*100:.2f}%")
        return results
    
    def predict(self, image):
        """Predict for a single image"""
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        predictions = self.model.predict(image, verbose=0)
        return predictions[0]
    
    def load_model(self, model_path):
        """Load saved model"""
        self.model = keras.models.load_model(model_path)
        print(f"✓ Model loaded from {model_path}")
        
    def save_model(self, model_path):
        """Save the model"""
        self.model.save(model_path)
        print(f"✓ Model saved to {model_path}")
    
    def summary(self):
        """Display model architecture"""
        return self.model.summary()