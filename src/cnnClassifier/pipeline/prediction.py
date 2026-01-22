import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        
    def predict(self):
        # Load model
        model = load_model(os.path.join("artifacts", "training", "model.h5"))
        
        # Class labels (alphabetical order matching folder names)
        CLASS_LABELS = ["Normal", "Tumor"]
        
        # Load and preprocess image
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # Normalize
        
        # Predict
        result = np.argmax(model.predict(test_image), axis=1)
        prediction = CLASS_LABELS[result[0]]
        
        # Get confidence scores
        probabilities = model.predict(test_image)[0]
        confidence = float(probabilities[result[0]] * 100)
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "is_tumor": prediction == "Tumor",
            "probabilities": {
                "Normal": round(float(probabilities[0] * 100), 2),
                "Tumor": round(float(probabilities[1] * 100), 2)
            }
        }
