# 🌿 Leaf Disease Detection Using Deep Learning

This project is a deep learning system that identifies plant leaf diseases from images using Convolutional Neural Networks (CNN). It was built using TensorFlow and Keras and trained on the PlantVillage dataset to classify 25 different leaf disease categories. The model achieves 85–90% accuracy and includes a prediction script for testing new images.

---

## 🔧 Technologies Used

- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Google Colab  
- HTML (for basic GUI)

---

## 📁 Project Files

| File Name         | Description |
|------------------|-------------|
| `cnn_train.py`    | Builds, trains, and saves the CNN model |
| `predict.py`      | Loads the trained model and predicts disease from a new image |
| `model1.json`     | Saved model architecture |
| `model1.h5`       | Saved model weights |
| `requirements.txt`| (Optional) Python dependencies (tensorflow, keras, opencv-python, numpy)

---

## 📂 Dataset

Use the [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)

### Folder Structure:
dataset/
├── train/
│ ├── Apple___Apple_scab/
│ ├── Tomato___Late_blight/
│ └── ... (other classes)
└── test/
├── Apple___Apple_scab/
└── ...

---

## 🚀 How to Run This Project

### Step 1: Install Required Libraries

```bash
pip install tensorflow keras opencv-python numpy
Step 2: Train the CNN Model
python cnn_train.py
Loads images from dataset/train and dataset/test

Applies data augmentation

Trains CNN with batch normalization and dropout

Saves model to model1.json and model1.h5
python predict.py
📷 Sample Prediction Output (Optional)
🔍 Prediction: Tomato___Late_blight
📊 Confidence Score: 0.932
