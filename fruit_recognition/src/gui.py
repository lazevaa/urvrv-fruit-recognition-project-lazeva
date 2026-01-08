"""
GUI Application for Fruit Recognition and Ripeness Detection
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from pathlib import Path

class FruitRecognitionGUI:
    def __init__(self, root, model, data_loader, color_analyzer, ripeness_detector):
        self.root = root
        self.model = model
        self.data_loader = data_loader
        self.color_analyzer = color_analyzer
        self.ripeness_detector = ripeness_detector
        
        self.root.title("🍎 Fruit Recognition & Ripeness Detector")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        self.current_image = None
        self.current_image_path = None
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI widgets"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title = tk.Label(
            title_frame,
            text="🍎 Fruit Recognition & Ripeness Detector",
            font=("Arial", 24, "bold"),
            bg='#2c3e50',
            fg='white'
        )
        title.pack(pady=20)
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Image display
        left_frame = tk.Frame(main_frame, bg='white', relief=tk.RAISED, borderwidth=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Image label
        self.image_label = tk.Label(left_frame, bg='white', text="No image loaded", 
                                   font=("Arial", 12), fg='gray')
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        # Buttons frame
        button_frame = tk.Frame(left_frame, bg='white')
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Load Image button
        load_btn = tk.Button(
            button_frame,
            text="📁 Load Image",
            command=self.load_image,
            font=("Arial", 12, "bold"),
            bg='#3498db',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        load_btn.pack(side=tk.LEFT, padx=5)
        
        # Analyze button
        self.analyze_btn = tk.Button(
            button_frame,
            text="🔍 Analyze",
            command=self.analyze_image,
            font=("Arial", 12, "bold"),
            bg='#27ae60',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        # Analyze Folder button
        folder_btn = tk.Button(
            button_frame,
            text="📂 Analyze Folder",
            command=self.analyze_folder,
            font=("Arial", 12, "bold"),
            bg='#9b59b6',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        folder_btn.pack(side=tk.LEFT, padx=5)
        
        # Right panel - Results
        right_frame = tk.Frame(main_frame, bg='white', relief=tk.RAISED, borderwidth=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Results title
        results_title = tk.Label(
            right_frame,
            text="📊 Analysis Results",
            font=("Arial", 16, "bold"),
            bg='white',
            fg='#2c3e50'
        )
        results_title.pack(pady=15)
        
        # Results text area with scrollbar
        results_frame = tk.Frame(right_frame, bg='white')
        results_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)
        
        scrollbar = tk.Scrollbar(results_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_text = tk.Text(
            results_frame,
            font=("Courier", 10),
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            bg='#ecf0f1',
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.results_text.yview)
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            bg='#34495e',
            fg='white',
            font=("Arial", 10),
            anchor=tk.W,
            padx=10
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def load_image(self):
        """Load an image file"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load and display image
                image = Image.open(file_path)
                self.current_image_path = file_path
                
                # Resize for display
                display_size = (400, 400)
                image.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(image)
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo
                
                # Load image for processing
                self.current_image = cv2.imread(file_path)
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                
                # Enable analyze button
                self.analyze_btn.config(state=tk.NORMAL)
                
                self.status_bar.config(text=f"Loaded: {os.path.basename(file_path)}")
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "Image loaded successfully!\nClick 'Analyze' to identify the fruit.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {str(e)}")
    
    def analyze_image(self):
        """Analyze the loaded image"""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first!")
            return
        
        try:
            self.status_bar.config(text="Analyzing...")
            self.root.update()
            
            # Prepare image for model
            img_resized = cv2.resize(self.current_image, (100, 100))
            img_normalized = img_resized.astype('float32') / 255.0
            
            # Get prediction
            predictions = self.model.predict(img_normalized)
            
            # Get top 3 predictions
            top_3_idx = np.argsort(predictions)[-3:][::-1]
            top_3_probs = predictions[top_3_idx]
            top_3_classes = [self.data_loader.get_class_name(idx) for idx in top_3_idx]
            
            # Get color analysis
            color_info = self.color_analyzer.analyze_color_distribution(self.current_image)
            
            # Get ripeness analysis
            predicted_fruit = top_3_classes[0]
            ripeness_info = self.ripeness_detector.detect_ripeness(self.current_image, predicted_fruit)
            
            # Display results
            self.display_results(top_3_classes, top_3_probs, color_info, ripeness_info)
            
            self.status_bar.config(text="Analysis complete!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during analysis: {str(e)}")
            self.status_bar.config(text="Error occurred")
    
    def display_results(self, classes, probabilities, color_info, ripeness_info):
        """Display analysis results"""
        self.results_text.delete(1.0, tk.END)
        
        # Title
        self.results_text.insert(tk.END, "="*50 + "\n", "title")
        self.results_text.insert(tk.END, "🍎 FRUIT CLASSIFICATION RESULTS\n", "title")
        self.results_text.insert(tk.END, "="*50 + "\n\n", "title")
        
        # Top prediction
        self.results_text.insert(tk.END, "🎯 PREDICTED FRUIT:\n", "header")
        self.results_text.insert(tk.END, f"   {classes[0]}\n", "fruit")
        self.results_text.insert(tk.END, f"   Confidence: {probabilities[0]*100:.2f}%\n\n", "confidence")
        
        # Top 3 predictions
        self.results_text.insert(tk.END, "📊 TOP 3 PREDICTIONS:\n", "header")
        for i, (cls, prob) in enumerate(zip(classes, probabilities), 1):
            bar = "█" * int(prob * 20)
            self.results_text.insert(tk.END, f"   {i}. {cls}\n")
            self.results_text.insert(tk.END, f"      [{bar:<20}] {prob*100:.2f}%\n\n")
        
        # Color analysis
        self.results_text.insert(tk.END, "🎨 COLOR ANALYSIS:\n", "header")
        avg_rgb = color_info['average_rgb']
        self.results_text.insert(tk.END, f"   Average RGB: ({avg_rgb[0]}, {avg_rgb[1]}, {avg_rgb[2]})\n")
        self.results_text.insert(tk.END, f"   Dominant Color: {color_info['color_name'].title()}\n\n")
        
        # Ripeness analysis
        self.results_text.insert(tk.END, "🍌 RIPENESS ANALYSIS:\n", "header")
        self.results_text.insert(tk.END, f"   Status: {ripeness_info['emoji']} {ripeness_info['level'].upper().replace('_', ' ')}\n", "ripeness")
        self.results_text.insert(tk.END, f"   Ripeness: {ripeness_info['percentage']}%\n")
        self.results_text.insert(tk.END, f"\n   💡 {ripeness_info['description']}\n\n")
        
        # Configure tags for formatting
        self.results_text.tag_config("title", font=("Arial", 12, "bold"), foreground="#2c3e50")
        self.results_text.tag_config("header", font=("Arial", 11, "bold"), foreground="#27ae60")
        self.results_text.tag_config("fruit", font=("Arial", 14, "bold"), foreground="#e74c3c")
        self.results_text.tag_config("confidence", font=("Arial", 10), foreground="#3498db")
        self.results_text.tag_config("ripeness", font=("Arial", 11, "bold"), foreground="#f39c12")
    
    def analyze_folder(self):
        """Analyze all images in a folder"""
        folder_path = filedialog.askdirectory(title="Select folder with images")
        
        if not folder_path:
            return
        
        try:
            # Get all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(Path(folder_path).glob(f'*{ext}'))
                image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
            
            if not image_files:
                messagebox.showinfo("No Images", "No image files found in the selected folder.")
                return
            
            self.status_bar.config(text=f"Analyzing {len(image_files)} images...")
            self.root.update()
            
            # Analyze all images
            results_summary = {}
            
            for img_path in image_files:
                try:
                    # Load image
                    img = cv2.imread(str(img_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img, (100, 100))
                    img_normalized = img_resized.astype('float32') / 255.0
                    
                    # Get prediction
                    predictions = self.model.predict(img_normalized)
                    predicted_idx = np.argmax(predictions)
                    predicted_class = self.data_loader.get_class_name(predicted_idx)
                    
                    # Count occurrences
                    if predicted_class in results_summary:
                        results_summary[predicted_class] += 1
                    else:
                        results_summary[predicted_class] = 1
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
            
            # Display summary
            self.display_folder_results(results_summary, len(image_files))
            self.status_bar.config(text=f"Analyzed {len(image_files)} images")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error analyzing folder: {str(e)}")
    
    def display_folder_results(self, results, total_images):
        """Display folder analysis results"""
        self.results_text.delete(1.0, tk.END)
        
        self.results_text.insert(tk.END, "="*50 + "\n", "title")
        self.results_text.insert(tk.END, "📂 FOLDER ANALYSIS RESULTS\n", "title")
        self.results_text.insert(tk.END, "="*50 + "\n\n", "title")
        
        self.results_text.insert(tk.END, f"Total images analyzed: {total_images}\n\n", "header")
        
        self.results_text.insert(tk.END, "🍎 FRUIT DISTRIBUTION:\n\n", "header")
        
        # Sort by count
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        for fruit, count in sorted_results:
            percentage = (count / total_images) * 100
            bar = "█" * int(percentage / 2)
            self.results_text.insert(tk.END, f"  {fruit}:\n")
            self.results_text.insert(tk.END, f"    Count: {count} ({percentage:.1f}%)\n")
            self.results_text.insert(tk.END, f"    [{bar:<50}]\n\n")
        
        # Configure tags
        self.results_text.tag_config("title", font=("Arial", 12, "bold"), foreground="#2c3e50")
        self.results_text.tag_config("header", font=("Arial", 11, "bold"), foreground="#27ae60")