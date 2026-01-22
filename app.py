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

# Global model variable
model = None


def download_model():
    """Download model from Google Drive if not exists"""
    model_path = "artifacts/training/model.h5"
    
    if not os.path.exists(model_path):
        print("📥 Downloading model from Google Drive...")
        os.makedirs("artifacts/training", exist_ok=True)
        
        file_id = "146vCH9kMZ7m6jVx7kGg2yKxwVBKr6vmt"
        url = f"https://drive.google.com/uc?id={file_id}"
        
        gdown.download(url, model_path, quiet=False)
        print("✅ Model downloaded successfully!")
    else:
        print("✅ Model already exists")
    
    return model_path


def load_model():
    """Load the trained model"""
    global model
    # Import tensorflow here to avoid loading at startup
    import tensorflow as tf
    
    model_path = download_model()
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("✅ Model loaded successfully!")
    return model


def decode_image(imgstring, filename):
    """Decode base64 image and save to file"""
    imgdata = base64.b64decode(imgstring)
    with open(filename, 'wb') as f:
        f.write(imgdata)


def predict_image(filename):
    """Make prediction on the image"""
    global model
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    
    if model is None:
        load_model()
    
    # Class labels
    CLASS_LABELS = ["Normal", "Tumor"]
    
    # Load and preprocess image
    test_image = image.load_img(filename, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255.0
    
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
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    # Pre-load model
    load_model()
    
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
