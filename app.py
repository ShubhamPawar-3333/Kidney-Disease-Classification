from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline
import os
import gdown

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


def download_model():
    """Download model from Google Drive if not exists"""
    model_path = "artifacts/training/model.h5"
    
    if not os.path.exists(model_path):
        print("📥 Downloading model from Google Drive...")
        os.makedirs("artifacts/training", exist_ok=True)
        
        # Replace this with your model's Google Drive file ID
        # Upload your model.h5 to Google Drive and get the shareable link
        # Example: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
        file_id = "146vCH9kMZ7m6jVx7kGg2yKxwVBKr6vmt"  # TODO: Replace with actual ID
        url = f"https://drive.google.com/uc?id={file_id}"
        
        gdown.download(url, model_path, quiet=False)
        print("✅ Model downloaded successfully!")
    else:
        print("✅ Model already exists")


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    return "Training done successfully!"


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)
        result = clApp.classifier.predict()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=['GET'])
@cross_origin()
def health():
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    # Download model on startup
    download_model()
    
    clApp = ClientApp()
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
