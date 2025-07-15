# 🍃 Leaf Disease Detection Using Machine Learning

A machine learning project aimed at detecting and classifying leaf diseases to assist farmers in early identification and treatment, ultimately improving crop yield and agricultural efficiency.

## 🧠 Objective

To develop a deep learning model capable of accurately identifying multiple types of leaf diseases from images, enabling timely diagnosis and intervention for farmers.

---

## ⚙️ Tech Stack

- **Programming Language**: Python  
- **Libraries & Frameworks**:
  - TensorFlow
  - Keras
  - OpenCV
  - NumPy
- **Others**:
  - Google Colab (for GPU training)
  - HTML (for simple GUI interface)

---

## 🔍 Features

- Preprocessing of over 1,500 leaf images using OpenCV (grayscale, resizing, normalization)
- CNN model architecture built using TensorFlow and Keras
- Achieved ~90% accuracy in classification across various leaf diseases
- Applied data augmentation (rotation, flipping, zooming) to increase model generalization
- Evaluation metrics: confusion matrix, accuracy, loss visualization
- Simple web-based HTML interface for displaying predictions

---

## 📁 Folder Structure

```bash
Leaf-Disease-Detection/
├── dataset/
│   ├── train/
│   ├── test/
├── model/
│   └── leaf_disease_model.h5
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate.py
├── gui/
│   └── index.html
├── README.md
└── requirements.txt
git clone https://github.com/KAVINPRABHAKAR/Leaf-Disease-Detection-Using-Deep-Learning.git
cd Leaf-Disease-Detection-Using-Deep-Learning
pip install -r requirements.txt
python src/train_model.py
python src/train_model.py
📊 Results
Accuracy: ~90%

Model Evaluation: Plotted confusion matrix and loss/accuracy graphs

Performance: Optimized with data augmentation and trained on GPU via Google Colab

