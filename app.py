"""DINOv2 Wildlife Interface - upload image -> species prediction.
Run: pip install gradio transformers torch torchvision pillow
     python app.py  (put unzipped dinov2-wildlife-finetuned/ next to this file)
"""
import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, Dinov2ForImageClassification
import torchvision.transforms as T

MODEL_DIR = os.environ.get("MODEL_DIR", "./dinov2-wildlife-finetuned")
IMG_SIZE = 224
THRESHOLD_DEFAULT = 0.5

print(f"Loading model from {MODEL_DIR} ...")
processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
model = Dinov2ForImageClassification.from_pretrained(MODEL_DIR)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Loaded on {device}. Classes: {model.config.num_labels}")

# robust id2label (keys may be str or int)
raw = model.config.id2label
id2label = {int(k): v for k, v in raw.items()}
CLASS_NAMES = [id2label[i] for i in range(len(id2label))]

def build_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

TF = build_transform()

def predict(img: Image.Image, threshold: float = THRESHOLD_DEFAULT):
    if img is None:
        return {}, "Upload an image"
    img = img.convert("RGB")
    x = TF(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(pixel_values=x).logits[0]
        probs = torch.sigmoid(logits).cpu().numpy()
    # full distribution for Label component
    conf = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))}
    kept = [(CLASS_NAMES[i], float(probs[i])) for i in range(len(probs)) if probs[i] >= threshold]
    kept.sort(key=lambda t: -t[1])
    top5 = sorted(conf.items(), key=lambda t: -t[1])[:5]
    if not kept:
        text = f"### No species above {threshold:.2f}\n\nTop-5:\n" + "\n".join([f"- {n}: {s:.3f}" for n, s in top5])
    else:
        text = f"### Predicted: {', '.join([n for n,_ in kept])}\n\n" + "\n".join([f"- {n}: {s:.3f}" for n, s in kept])
        text += "\n\nTop-5:\n" + "\n".join([f"- {n}: {s:.3f}" for n, s in top5])
    # Gradio Label wants top confusing dict; return filtered to threshold or top5
    label_out = {n: s for n, s in kept} if kept else dict(top5)
    return label_out, text

if __name__ == "__main__":
    import gradio as gr
    with gr.Blocks(title="Wildlife DINOv2 Classifier (54 species)") as demo:
        gr.Markdown("# 🦁 Wildlife Conservation — DINOv2 Classifier\nUpload an animal photo, get species + confidence.")
        with gr.Row():
            inp = gr.Image(type="pil", label="Upload image")
            thr = gr.Slider(0.1, 0.9, step=0.05, value=THRESHOLD_DEFAULT, label="Threshold")
        out_label = gr.Label(num_top_classes=5, label="Prediction")
        out_text = gr.Markdown()
        btn = gr.Button("Predict")
        btn.click(predict, [inp, thr], [out_label, out_text])
        # live update on upload too
        inp.change(predict, [inp, thr], [out_label, out_text])
    demo.launch(share=False)
