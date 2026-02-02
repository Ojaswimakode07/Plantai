"""
Plant disease prediction using ResNet34
Loads weights from Models/plantDisease-resnet34.pth
"""

import io
import os

CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

_model = None


def _get_model():
    global _model
    if _model is not None:
        return _model

    import torch
    import torch.nn as nn
    from torchvision import models

    class PlantDiseaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = models.resnet34(weights=None)
            self.network.fc = nn.Linear(self.network.fc.in_features, len(CLASSES))

        def forward(self, x):
            return self.network(x)

    _model = PlantDiseaseModel()
    _model.eval()

    base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, "Models", "plantDisease-resnet34.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    state = torch.load(model_path, map_location="cpu")
    _model.load_state_dict(state)

    return _model


def predict_image(img_bytes):
    from PIL import Image
    import torch
    import torchvision.transforms as transforms

    model = _get_model()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    xb = transform(image).unsqueeze(0)

    with torch.no_grad():
        preds = model(xb)
        _, idx = torch.max(preds, 1)

    return CLASSES[idx.item()]
