# 🦁 Wildlife Conservation — DINOv2 Multi-Label Species Classifier

 Self-Supervised Deep Learning for Wildlife Conservation  
> Fine-tuned `facebook/dinov2-small` (Vision Transformer) to classify **54 wildlife species** from a single image using multi-label classification.

---

## 📌 Table of Contents
- [Demo](#-demo)
- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [How to Run Locally](#-how-to-run-locally)
- [Project Structure](#-project-structure)
- [Supported Species](#-supported-species-54)

---

## 🎬 Demo

Upload any animal photo → get predicted species + confidence scores via a Gradio web interface.

```
Upload image → DINOv2 Encoder → Linear Classifier → Species predictions (with confidence)
```

---

## 📖 Project Overview

This project applies **self-supervised deep learning** to wildlife conservation by fine-tuning Meta's **DINOv2** vision transformer on a curated wildlife image dataset. The model performs **multi-label image classification**, meaning it can detect multiple species in a single image simultaneously.

Key highlights:
- 🧠 **Foundation model**: `facebook/dinov2-small` (21M parameters)
- 🏷️ **Task**: Multi-label classification across 54 animal species
- 🎯 **Threshold-based inference**: Configurable confidence threshold (default: 0.5)
- 🌐 **Web interface**: Gradio-powered UI for easy image upload and prediction

---

## 🏗️ Architecture

```
Input Image (any size)
        │
        ▼
┌─────────────────────────────────────────┐
│         Preprocessing Pipeline          │
│  Resize → 224×224                       │
│  Normalize (ImageNet mean/std)          │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│        DINOv2-Small Backbone            │
│  Vision Transformer (ViT-S/14)          │
│  • Patch size: 14×14                    │
│  • Hidden size: 384                     │
│  • 12 Transformer layers                │
│  • 6 attention heads                    │
│  • MLP ratio: 4                         │
│  • Activation: GELU                     │
│  • Pre-trained via self-supervised DINO │
└─────────────────────────────────────────┘
        │
        ▼  [CLS] token embedding (384-dim)
┌─────────────────────────────────────────┐
│         Linear Classification Head      │
│  Linear(384 → 54)                       │
│  Sigmoid activation (multi-label)       │
└─────────────────────────────────────────┘
        │
        ▼
  54 confidence scores (0.0 – 1.0)
  Species predicted if score ≥ threshold
```

### Training Configuration
| Parameter | Value |
|---|---|
| Base model | `facebook/dinov2-small` |
| Image size | 224 × 224 |
| Epochs | 12 |
| Batch size (train) | 32 |
| Batch size (eval) | 64 |
| Learning rate | 5e-5 |
| Weight decay | 0.01 |
| Warmup ratio | 10% |
| Backbone | Full fine-tune (not frozen) |
| Loss | Binary Cross-Entropy (multi-label) |
| Inference threshold | 0.5 (sigmoid) |
| Trained on | Kaggle (T4/P100 GPU) |

---

## 📦 Dataset

**Source:** [Wildlife Dataset on Kaggle](https://www.kaggle.com/datasets/banuprasadb/wildlife-dataset)

The dataset contains labeled images of **54 wildlife species** in YOLO detection format (bounding box `.txt` files), which were converted to **multi-hot classification labels** during training.

### Train / Validation / Test Split

| Split | Folder | Purpose |
|---|---|---|
| **Train** | `dataset/train/` | Model training |
| **Validation** | `dataset/valid/` | Hyperparameter tuning & early stopping |
| **Test** | `dataset/test/` | Final evaluation |

> ⚠️ The dataset is **not included** in this repository due to its large size (~5.5 GB).  
> Download it from the Kaggle link above and place it in a `dataset/` folder at the project root.

### Label Format
- Original: YOLO `.txt` files (class_id + bounding box coordinates)
- Converted to: **multi-hot binary vectors** of length 54 for classification

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Deep Learning Framework** | PyTorch |
| **Model & Tokenizer** | HuggingFace Transformers |
| **Base Model** | `facebook/dinov2-small` (ViT-S/14) |
| **Image Processing** | Torchvision, Pillow |
| **Training Utilities** | HuggingFace `Trainer`, `evaluate`, `accelerate` |
| **Web Interface** | Gradio |
| **Model Format** | SafeTensors (`.safetensors`) |
| **Training Environment** | Kaggle Notebooks (GPU) |
| **Label Parsing** | scikit-learn, PyYAML |

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.9+
- pip
- (Optional but recommended) NVIDIA GPU with CUDA

### Step 1 — Clone the repository
```bash
git clone https://github.com/Deepali1237/wildlife-conservation-cse471.git
cd wildlife-conservation-cse471
```

### Step 2 — Install dependencies
```bash
pip install gradio transformers torch torchvision pillow
```

### Step 3 — Download the model weights
The model weights (`model.safetensors`) are stored via **Git LFS**.  
If they didn't download automatically, run:
```bash
git lfs pull
```
Or manually download the `dinov2-wildlife-finetuned/` folder and place it in the project root.

### Step 4 — Launch the Gradio web app
```bash
python app.py
```
Then open your browser at: **http://127.0.0.1:7860**

---

### Alternative: Command-line prediction (no Gradio)
```bash
python dinov2-wildlife-finetuned/predict.py \
  --model ./dinov2-wildlife-finetuned \
  --img your_animal_photo.jpg \
  --threshold 0.5
```

Example output:
```
Predicted: ['Elephant', 'Giraffe']
Top-5:
  Elephant: 0.912
  Giraffe: 0.783
  Lion: 0.234
  Zebra: 0.187
  Rhinoceros: 0.102
```

---

### Using GPU vs CPU
The app automatically detects and uses GPU if available:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## 📁 Project Structure

```
wildlife-conservation/
│
├── app.py                          # Gradio web interface (main entry point)
├── dinov2_wildlife_kaggle.ipynb    # Training notebook (run on Kaggle)
├── .gitignore
├── README.md
│
└── dinov2-wildlife-finetuned/      # Fine-tuned model artifacts
    ├── model.safetensors           # Model weights (~84 MB, via Git LFS)
    ├── config.json                 # Model architecture config
    ├── preprocessor_config.json    # Image preprocessing config
    ├── id2label.json               # Class index → species name mapping
    ├── label2id.json               # Species name → class index mapping
    ├── training_args.bin           # Training arguments used
    ├── predict.py                  # Standalone CLI prediction script
    └── README_RUN.txt              # Quick-start instructions
```

---

## 🐾 Supported Species (54)

| # | Species | # | Species | # | Species |
|---|---|---|---|---|---|
| 0 | Zebra | 19 | Fox | 38 | Woodpecker |
| 1 | Lion | 20 | Tortoise | 39 | Sheep |
| 2 | Leopard | 21 | Frog | 40 | Magpie |
| 3 | Cheetah | 22 | Kangaroo | 41 | Ostrich |
| 4 | Tiger | 23 | Deer | 42 | Jaguar |
| 5 | Bear | 24 | Eagle | 43 | Hedgehog |
| 6 | Butterfly | 25 | Monkey | 44 | Turkey |
| 7 | Canary | 26 | Snake | 45 | Raccoon |
| 8 | Crocodile | 27 | Owl | 46 | Worm |
| 9 | Bull | 28 | Swan | 47 | Harbor Seal |
| 10 | Camel | 29 | Goat | 48 | Panda |
| 11 | Centipede | 30 | Rabbit | 49 | Red Panda |
| 12 | Caterpillar | 31 | Giraffe | 50 | Otter |
| 13 | Duck | 32 | Goose | 51 | Lynx |
| 14 | Squirrel | 33 | Polar Bear | 52 | Scorpion |
| 15 | Spider | 34 | Raven | 53 | Koala |
| 16 | Ladybug | 35 | Hippopotamus | | |
| 17 | Elephant | 36 | Brown Bear | | |
| 18 | Horse | 37 | Rhinoceros | | |

---

## 📄 License
This project is for academic purposes (CSE471).  
Dataset credits: [banuprasadb on Kaggle](https://www.kaggle.com/datasets/banuprasadb/wildlife-dataset)  
Base model credits: [Meta AI — DINOv2](https://github.com/facebookresearch/dinov2)
