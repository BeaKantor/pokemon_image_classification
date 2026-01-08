# Pokémon Image Classification using Machine Learning and Deep Learning

## Project Overview
This project focuses on classifying Pokémon images into their respective species using:
- traditional Machine Learning models
- Convolutional Neural Networks (CNN)
- Transfer Learning with MobileNetV2

The goal is to compare classical ML approaches with modern Deep Learning techniques on a real-world image dataset.

---

##  Dataset
- Source: Kaggle – Pokémon Classification Dataset
- ~8000 images
- 150+ classes
- Images are organized in folders by Pokémon name

 The dataset is not included in this repository due to size limitations.

---

##  Models Used

### 1. RandomForest (Baseline ML)
- Images resized to 32×32
- Flattened into 1D vectors
- Parameters:
  - `n_estimators = 100`
  - `max_depth = 12`
- Result: Strong overfitting, poor generalization

### 2. SVC (Support Vector Classifier)
- RBF kernel
- Trained on a subset for efficiency
- Result: Low validation accuracy on image data

### 3. CNN (from scratch)
- 3 Convolutional layers + MaxPooling
- Dense + Dropout
- Data augmentation applied
- Result: Moderate performance (~52% accuracy)

### 4. MobileNetV2 (Transfer Learning)
- Pretrained on ImageNet
- Feature extraction + fine-tuning
- Learning rate: `1e-4`
- Best result: ~83% validation accuracy

---

##  Results Summary

| Model | Validation Accuracy |
|------|---------------------|
| RandomForest | ~0.31 |
| SVC | ~0.20 |
| CNN | ~0.52 |
| **MobileNetV2** | **~0.83** |

---

##  Conclusion
- Classical ML models are not suitable for complex image classification tasks.
- CNNs significantly improve performance by learning visual features.
- Transfer Learning with MobileNetV2 provides the best results with limited data.

---

##  Technologies
- Python
- TensorFlow / Keras
- Scikit-learn
- Matplotlib / Seaborn
- Google Colab

---

##  Author
- Project developed for educational purposes
