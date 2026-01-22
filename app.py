# Load TensorFlow at startup (not per-request)
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
import numpy as np
import base64
import os
import gdown

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

# ============ LOAD MODEL AT STARTUP ============
MODEL_PATH = "artifacts/training/model.h5"
GDRIVE_FILE_ID = "146vCH9kMZ7m6jVx7kGg2yKxwVBKr6vmt"
CLASS_LABELS = ["Normal", "Tumor"]

# Global model - loaded once at startup
model = None


def download_model_if_needed():
    """Download model from Google Drive if not exists"""
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")
        os.makedirs("artifacts/training", exist_ok=True)
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("✅ Model downloaded successfully!")


def load_model_at_startup():
    """Load model once at startup"""
    global model
    download_model_if_needed()
    
    print("🔄 Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    # Warm up the model with a dummy prediction
    dummy = np.zeros((1, 224, 224, 3))
    model.predict(dummy, verbose=0)
    print("✅ Model loaded and warmed up!")


# ============ LIGHTWEIGHT INFERENCE ============
def decode_image(imgstring, filename):
    """Decode base64 image and save to file"""
    imgdata = base64.b64decode(imgstring)
    with open(filename, 'wb') as f:
        f.write(imgdata)


def predict_image(filename):
    """Lightweight prediction - model already loaded"""
    # Load and preprocess image
    test_image = keras_image.load_img(filename, target_size=(224, 224))
    test_image = keras_image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255.0
    
    # Predict (model already in memory)
    probabilities = model.predict(test_image, verbose=0)[0]
    result_idx = np.argmax(probabilities)
    prediction = CLASS_LABELS[result_idx]
    confidence = float(probabilities[result_idx] * 100)
    
    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "is_tumor": prediction == "Tumor",
        "probabilities": {
            "Normal": round(float(probabilities[0] * 100), 2),
            "Tumor": round(float(probabilities[1] * 100), 2)
        }
    }


# ============ ROUTES ============
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        image_data = request.json['image']
        filename = "inputImage.jpg"
        decode_image(image_data, filename)
        result = predict_image(filename)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=['GET'])
@cross_origin()
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})


# ============ STARTUP ============
# Load model when app starts (not per-request)
print("🚀 Starting Kidney Disease Classifier...")
load_model_at_startup()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port, debug=False)
