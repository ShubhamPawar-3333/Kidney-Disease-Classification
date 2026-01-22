"""
Script to upload trained model to HuggingFace Hub.
Run this after training a new model.

Usage:
    python scripts/upload_model_to_hf.py

Prerequisites:
    1. Install huggingface_hub: pip install huggingface_hub
    2. Login to HuggingFace: huggingface-cli login
    3. Create a model repo: AIenthusSP/kidney-disease-model
"""

from huggingface_hub import HfApi, create_repo
import os

# Configuration
HF_REPO_ID = "AIenthusSP/kidney-disease-model"
MODEL_PATH = "artifacts/training/model.h5"


def upload_model():
    """Upload trained model to HuggingFace Hub"""
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        print("   Run training first: python main.py")
        return False
    
    print(f"📦 Uploading model to HuggingFace Hub: {HF_REPO_ID}")
    
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        create_repo(HF_REPO_ID, repo_type="model", exist_ok=True)
        print(f"✅ Repository ready: {HF_REPO_ID}")
    except Exception as e:
        print(f"ℹ️ Repo status: {e}")
    
    # Upload model file
    try:
        api.upload_file(
            path_or_fileobj=MODEL_PATH,
            path_in_repo="model.h5",
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message="Update trained model"
        )
        print(f"✅ Model uploaded successfully!")
        print(f"   View at: https://huggingface.co/{HF_REPO_ID}")
        return True
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False


if __name__ == "__main__":
    upload_model()
