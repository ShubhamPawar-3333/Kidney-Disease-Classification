# VGG16 Kidney Disease Classification Model Architecture

## Model Overview

```mermaid
flowchart TB
    subgraph Input
        A[Input Image<br/>224 × 224 × 3]
    end
    
    subgraph VGG16["VGG16 Base (Frozen)"]
        B1[Conv Block 1<br/>64 filters]
        B2[Conv Block 2<br/>128 filters]
        B3[Conv Block 3<br/>256 filters]
        B4[Conv Block 4<br/>512 filters]
        B5[Conv Block 5<br/>512 filters]
    end
    
    subgraph Custom["Custom Head (Trainable)"]
        F[Flatten<br/>7 × 7 × 512 = 25,088]
        D[Dense<br/>2 units, Softmax]
    end
    
    subgraph Output
        O[Prediction<br/>Normal / Tumor]
    end
    
    A --> B1 --> B2 --> B3 --> B4 --> B5
    B5 --> F --> D --> O
    
    style Input fill:#e1f5fe
    style Output fill:#c8e6c9
    style VGG16 fill:#fff3e0
    style Custom fill:#f3e5f5
```

---

## Detailed Layer Structure

| Block | Layers | Output Shape | Parameters |
|-------|--------|--------------|------------|
| **Input** | InputLayer | (224, 224, 3) | 0 |
| **Block 1** | Conv2D × 2 + MaxPool | (112, 112, 64) | 38,720 |
| **Block 2** | Conv2D × 2 + MaxPool | (56, 56, 128) | 221,440 |
| **Block 3** | Conv2D × 3 + MaxPool | (28, 28, 256) | 1,475,328 |
| **Block 4** | Conv2D × 3 + MaxPool | (14, 14, 512) | 5,899,776 |
| **Block 5** | Conv2D × 3 + MaxPool | (7, 7, 512) | 5,899,776 |
| **Flatten** | Flatten | (25088,) | 0 |
| **Dense** | Dense + Softmax | (2,) | 50,178 |

---

## Model Summary

| Property | Value |
|----------|-------|
| **Base Model** | VGG16 (ImageNet pretrained) |
| **Total Parameters** | ~14.7 Million |
| **Trainable Parameters** | 50,178 (Dense layer only) |
| **Frozen Parameters** | ~14.7 Million (VGG16 base) |
| **Input Size** | 224 × 224 × 3 |
| **Output Classes** | 2 (Normal, Tumor) |
| **Optimizer** | SGD (lr=0.001) |
| **Loss Function** | Categorical Crossentropy |

---

## Data Flow

```
CT Scan Image (224×224×3)
         ↓
   VGG16 Feature Extractor (Frozen)
         ↓
   Feature Map (7×7×512)
         ↓
   Flatten (25,088 features)
         ↓
   Dense Layer (2 neurons)
         ↓
   Softmax Activation
         ↓
   [Normal: 0.92, Tumor: 0.08]
```
