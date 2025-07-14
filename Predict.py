# predict.py

import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json

# Load model
with open('model1.json', 'r') as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model1.h5")
print("✅ Loaded model from disk")

# Disease class labels
labels = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___Healthy",
    "Corn_(maize)___Cercospora_leaf_spot", "Corn_(maize)___Common_rust_", "Corn_(maize)___Healthy",
    "Corn_(maize)___Northern_Leaf_Blight", "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Healthy", "Grape___Leaf_blight", "Potato___Early_blight", "Potato___Healthy",
    "Potato___Late_blight", "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Healthy",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites", "Tomato___Target_Spot", "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___Mosaic_virus"
]

# Load and preprocess test image
img_path = 'im_for_testing_purpose/a.scab.JPG'
test_image = image.load_img(img_path, target_size=(128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Predict
result = loaded_model.predict(test_image)
predicted_class = labels[np.argmax(result)]
confidence = np.max(result)

print("🔍 Prediction:", predicted_class)
print("📊 Confidence Score:", confidence)
