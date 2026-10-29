"""Run fine-tuned DINOv2 wildlife model anywhere.
Usage:
  pip install transformers torch pillow
  python predict.py --model ./dinov2-wildlife-finetuned --img elephant.jpg [--threshold 0.5]
"""
import argparse, json, os
import torch
from PIL import Image
from transformers import AutoImageProcessor, Dinov2ForImageClassification

def load(model_dir):
    proc = AutoImageProcessor.from_pretrained(model_dir)
    model = Dinov2ForImageClassification.from_pretrained(model_dir)
    model.eval()
    return proc, model

def predict(proc, model, img_path, threshold=0.5, img_size=224):
    import torchvision.transforms as T
    img = Image.open(img_path).convert("RGB")
    tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(),
                    T.Normalize(mean=proc.image_mean, std=proc.image_std)])
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(pixel_values=x).logits[0]
        probs = torch.sigmoid(logits).cpu().numpy()
    ids = (probs >= threshold).nonzero()[0].tolist()
    names = [model.config.id2label[str(i)] if str(i) in model.config.id2label else model.config.id2label[i] for i in ids]
    scored = sorted([(model.config.id2label.get(str(i), model.config.id2label.get(i, str(i))), float(probs[i])) for i in range(len(probs))], key=lambda t: -t[1])[:5]
    return names, probs, scored

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    proc, model = load(a.model)
    names, probs, scored = predict(proc, model, a.img, a.threshold)
    print("Predicted:", names if names else ["(none above threshold)"])
    print("Top-5:")
    [print(f"  {n}: {s:.3f}") for n, s in scored]
