# 🔄 Complete Project Pipeline Flow

## Quick Overview

```
Data → Training → MLflow → Upload to HF Hub → Deploy to HF Spaces → Web App
```

---

## 📋 Complete Sequential Flow

### PHASE 1: Development & Training (Local Machine)

#### Step 1: Data Ingestion
**File:** `src/cnnClassifier/pipeline/stage_01_data_ingestion.py`
**What happens:** Downloads/prepares kidney CT scan dataset
```bash
python main.py  # Runs all stages
```

#### Step 2: Prepare Base Model
**File:** `src/cnnClassifier/pipeline/stage_02_prepare_base_model.py`
**What happens:**
- Loads VGG16 pretrained on ImageNet
- Freezes base layers
- Adds custom Dense layer for 2 classes
**Output:** `artifacts/prepare_base_model/base_model_updated.h5`

#### Step 3: Model Training
**File:** `src/cnnClassifier/pipeline/stage_03_model_training.py`
**What happens:**
- Loads training data with augmentation
- Trains for 20 epochs
- Saves trained weights
**Output:** `artifacts/training/model.h5` ⬅️ **This is your trained model**

#### Step 4: Model Evaluation
**File:** `src/cnnClassifier/pipeline/stage_04_model_evaluation.py`
**What happens:**
- Evaluates on validation data
- Logs metrics to MLflow
- Saves scores
**Output:** `scores.json`, `mlruns/` (experiment logs)

---

### PHASE 2: Model Upload (Local → HuggingFace Hub)

#### Step 5: Upload Model to HuggingFace
**File:** `scripts/upload_model_to_hf.py`
**What happens:**
- Takes `artifacts/training/model.h5`
- Uploads to HuggingFace Model Hub
```bash
python scripts/upload_model_to_hf.py
```
**Output:** Model available at `https://huggingface.co/AIenthusSP/kidney-disease-model`

---

### PHASE 3: Deployment (GitHub → HuggingFace Spaces)

#### Step 6: Push Code to GitHub
```bash
git add .
git commit -m "Update"
git push origin main
```

#### Step 7: GitHub Action Syncs to HuggingFace
**File:** `.github/workflows/sync-to-hf.yml`
**What happens:**
- Triggers on push to `main`
- Copies `app.py`, `Dockerfile`, `requirements.txt`, `templates/`
- Creates HuggingFace-compatible README
- Pushes to HuggingFace Spaces

---

### PHASE 4: Runtime (HuggingFace Spaces)

#### Step 8: Docker Build
**File:** `Dockerfile`
**What happens:**
- Builds Docker container on HuggingFace
- Installs dependencies from `requirements.txt`
- Prepares the Flask app

#### Step 9: App Startup
**File:** `app.py`
**What happens:**
- Downloads model from HuggingFace Hub (or Google Drive fallback)
- Loads TensorFlow and model into memory
- Starts Flask server on port 7860

#### Step 10: User Prediction
**What happens:**
1. User uploads CT scan image on web UI
2. Frontend sends base64 image to `/predict` endpoint
3. Flask app preprocesses image (224x224, normalize)
4. TensorFlow model predicts (Normal/Tumor)
5. Returns probability + class to frontend
6. UI displays result

---

## 🗂️ Key Files & Their Purpose

### Configuration Files
| File | Purpose |
|------|---------|
| `config/config.yaml` | Artifact paths, data sources |
| `params.yaml` | Hyperparameters (epochs, batch size, etc.) |
| `dvc.yaml` | Pipeline stage definitions |
| `dvc.lock` | Pipeline checksums for reproducibility |

### Pipeline Components
| File | Purpose |
|------|---------|
| `src/cnnClassifier/components/data_ingestion.py` | Download & prepare data |
| `src/cnnClassifier/components/prepare_base_model.py` | Create VGG16 model |
| `src/cnnClassifier/components/model_training.py` | Train the model |
| `src/cnnClassifier/components/model_evaluation.py` | Evaluate & log metrics |

### Deployment Files
| File | Purpose |
|------|---------|
| `app.py` | Flask web application |
| `Dockerfile` | Docker configuration for HF Spaces |
| `templates/index.html` | Web UI |
| `.github/workflows/sync-to-hf.yml` | Auto-deploy to HuggingFace |
| `scripts/upload_model_to_hf.py` | Upload model to HF Hub |

---

## 🔁 Quick Reference Commands

### Training Workflow
```bash
# Full training pipeline
python main.py

# Or using DVC (smarter - skips unchanged stages)
dvc repro
```

### Upload Model
```bash
# Login to HuggingFace (one-time)
huggingface-cli login

# Upload trained model
python scripts/upload_model_to_hf.py
```

### Deploy Code Changes
```bash
git add .
git commit -m "Your message"
git push origin main
# GitHub Action auto-syncs to HuggingFace
```

### View Experiment Logs
```bash
mlflow ui
# Open http://localhost:5000
```

### Run Locally
```bash
python app.py
# Open http://localhost:7860
```

---

## 📊 Where to Look

| For This | Look Here |
|----------|-----------|
| Training logs | `logs/project_logs.log` |
| Model metrics | `scores.json` |
| Experiment tracking | `mlruns/` or `mlflow ui` |
| Trained model | `artifacts/training/model.h5` |
| Deployment status | GitHub Actions tab |
| App logs | HuggingFace Space → Logs tab |

---

## 🎯 Typical Workflow Summary

```
1. Make changes to code/params
2. Run: python main.py (or dvc repro)
3. Run: python scripts/upload_model_to_hf.py
4. Run: git add . && git commit && git push
5. App auto-updates on HuggingFace Spaces! 🚀
```
