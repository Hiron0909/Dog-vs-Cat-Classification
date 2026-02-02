import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request
import os

# --------------------
# CONFIG
# --------------------
MODEL_PATH = "model/dog_vs_cat_resnet18_professional.pth"
CLASS_NAMES = ["cat", "dog"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# LOAD MODEL
# --------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)   # IMPORTANT
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# --------------------
# TRANSFORMS
# --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------
# FLASK APP
# --------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            image = Image.open(file).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)

            prediction = CLASS_NAMES[pred.item()]
            confidence = round(conf.item() * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)
