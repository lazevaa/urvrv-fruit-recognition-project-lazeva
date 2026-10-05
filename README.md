# 🍎 Fruit Recognition & Ripeness Detector

Advanced CNN-based application for fruit classification and ripeness detection using deep learning.

## 📋 Project Overview

This project implements a Convolutional Neural Network (CNN) to:
- **Classify** 131 different types of fruits
- **Analyze colors** using computer vision
- **Detect ripeness** based on visual characteristics
- **Provide recommendations** for consumption

## 🎯 Features

- ✅ High-accuracy fruit classification (90%+ accuracy)
- ✅ Real-time ripeness detection for common fruits
- ✅ Color analysis (RGB, HSV, dominant colors)
- ✅ Batch processing of multiple images
- ✅ User-friendly GUI interface
- ✅ Detailed statistical analysis
- ✅ Visual training metrics and confusion matrices

## 📊 Dataset

This project uses the **Fruit-360** dataset from Kaggle:
- **Link**: https://www.kaggle.com/datasets/moltean/fruits
- **Size**: 90,000+ images
- **Classes**: 131 different fruits
- **Image size**: 100x100 pixels

## 🛠️ Installation

### 1. Clone or Download Project

```bash
# Create project directory
mkdir fruit_recognition
cd fruit_recognition
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- tensorflow==2.15.0
- keras==2.15.0
- numpy==1.24.3
- opencv-python==4.8.1.78
- matplotlib==3.8.0
- pillow==10.1.0
- scikit-learn==1.3.2
- seaborn==0.13.0
- pandas==2.1.3

### 3. Download Dataset

1. Go to https://www.kaggle.com/datasets/moltean/fruits
2. Download the dataset (archive.zip)
3. Extract it to your project folder
4. Rename the extracted folder to `dataset`

**Expected folder structure:**
```
fruit_recognition/
│
├── dataset/
│   ├── Training/
│   │   ├── Apple Braeburn/
│   │   ├── Apple Crimson Snow/
│   │   ├── Banana/
│   │   ├── Orange/
│   │   └── ... (131 folders total)
│   │
│   └── Test/
│       ├── Apple Braeburn/
│       ├── Banana/
│       └── ...
│
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── color_analyzer.py
│   ├── ripeness_detector.py
│   └── gui.py
│
├── models/ (created automatically)
├── results/ (created automatically)
├── train_model.py
├── main.py
└── requirements.txt
```

## 🚀 Usage

### Step 1: Train the Model

```bash
python train_model.py
```

**What happens:**
- Loads and preprocesses 90,000+ images
- Trains CNN model (20-30 minutes on GPU, 1-2 hours on CPU)
- Saves trained model to `models/fruit_model.h5`
- Generates training plots and confusion matrix in `results/`
- Creates classification report

**Expected output:**
```
📊 Step 1: Loading data...
✓ Loaded 131 fruit classes
✓ Training: 53879 images
✓ Validation: 13470 images
✓ Test: 22688 images

🏗️  Step 2: Building CNN model...
✓ CNN model built successfully!

🚀 Step 3: Training model...
Epoch 1/50
...
Epoch 25/50 (Early stopping)

📊 Test set results:
  - Accuracy: 92.45%
  - Top-3 Accuracy: 98.12%

✅ TRAINING COMPLETED SUCCESSFULLY!
```

### Step 2: Run the Application

```bash
python main.py
```

**GUI Features:**
1. **Load Image**: Select a fruit image
2. **Analyze**: Get classification and ripeness results
3. **Analyze Folder**: Process multiple images at once

## 🔍 How It Works

### 1. Image Classification

The CNN model architecture:
```
- Conv2D layers (32, 64, 128, 256 filters)
- Batch Normalization
- MaxPooling and Dropout
- Dense layers (512, 256)
- Softmax output (131 classes)
```

### 2. Color Analysis

- Extracts dominant colors using K-means clustering
- Calculates average RGB values
- Converts to HSV color space
- Identifies color names

### 3. Ripeness Detection

**Fruit-specific rules:**

**Banana:**
- 🟢 Green (g > r, g > b) → Unripe
- 🟡 Yellow (r > 180, g > 180, b < 120) → Ripe
- 🟤 Brown spots (darker colors) → Overripe

**Apple:**
- 🍏 Green (g > r) → Crisp and tart
- 🍎 Red (r > 150) → Sweet and juicy
- 🟤 Brown (mixed colors) → Overripe

**Avocado:**
- 🟢 Bright green (brightness > 100) → Unripe
- 🥑 Dark green (brightness < 100) → Ripe
- ⚫ Very dark (brightness < 50) → Overripe

**And many more fruits...**

## 📊 Example Results

### Single Image Analysis

```
🎯 PREDICTED FRUIT: Banana
   Confidence: 98.45%

📊 TOP 3 PREDICTIONS:
   1. Banana
      [████████████████████] 98.45%
   2. Apple Golden
      [█░░░░░░░░░░░░░░░░░░░] 1.23%
   3. Pear
      [░░░░░░░░░░░░░░░░░░░░] 0.32%

🎨 COLOR ANALYSIS:
   Average RGB: (245, 231, 85)
   Dominant Color: Yellow

🍌 RIPENESS ANALYSIS:
   Status: 🟡 RIPE
   Ripeness: 85%
   
   💡 Yellow banana - perfect for eating! Sweet and delicious.
```

### Folder Analysis

```
📂 FOLDER ANALYSIS RESULTS

Total images analyzed: 150

🍎 FRUIT DISTRIBUTION:

  Apple Red:
    Count: 45 (30.0%)
    [███████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]

  Banana:
    Count: 38 (25.3%)
    [████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]

  Orange:
    Count: 32 (21.3%)
    [██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
```

## 🎯 Supported Fruits

The model recognizes **131 types of fruits**, including:

**Popular fruits:**
- Apples (12 varieties)
- Bananas (2 varieties)
- Oranges (2 varieties)
- Grapes (2 varieties)
- Strawberries
- Mangoes
- Pineapples
- Watermelons
- And many more!

**Full list**: Check `models/class_mappings.json` after training

## 📈 Model Performance

**Training Metrics:**
- Training Accuracy: ~95%
- Validation Accuracy: ~93%
- Test Accuracy: ~92%
- Top-3 Accuracy: ~98%

**Performance files:**
- `results/training_history.png` - Accuracy/loss curves
- `results/confusion_matrix.png` - Top 20 classes confusion matrix
- `results/classification_report.txt` - Detailed metrics per class

## 🎨 Ripeness Detection Logic

### Color-Based Analysis

1. **Extract fruit mask** (remove background)
2. **Calculate color metrics:**
   - Dominant color (K-means)
   - Average RGB
   - HSV values
   - Brightness
3. **Apply fruit-specific rules**
4. **Determine ripeness level:**
   - Unripe (20-40%)
   - Slightly unripe (60-70%)
   - Ripe (80-90%)
   - Overripe (95%+)

## 🤝 Contributing

This is an academic project, but suggestions are welcome!

## 📄 License

This project is for educational purposes.

## 👤 Author

**Jovana Lazeva**  Computer Science student at FERI, University of Maribor
Course project: Uvod v Računalniški Vid in Razpoznavanje Vzorcev 2025/26

## 🙏 Acknowledgments

- Fruit-360 dataset by Horea Muresan and Mihai Oltean
- Kaggle for hosting the dataset
- TensorFlow and Keras teams

## 📞 Support

If you encounter issues:

1. **Dataset not found:**
   - Verify `dataset/Training/` and `dataset/Test/` exist
   - Check folder structure matches above

2. **Out of memory:**
   - Reduce batch_size in `train_model.py` (try 16 or 8)
   - Use fewer images for testing

3. **Model not found:**
   - Run `train_model.py` first
   - Check `models/fruit_model.h5` exists

4. **Low accuracy:**
   - Train for more epochs
   - Adjust learning rate
   - Add more data augmentation

## 🚀 Future Improvements

- [ ] Mobile app version
- [ ] Real-time camera detection
- [ ] Multi-fruit detection in single image
- [ ] Nutritional information integration
- [ ] Price prediction based on ripeness
- [ ] Database of ripening tips

---
