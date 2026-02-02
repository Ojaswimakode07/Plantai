"""
Plant disease prediction using ResNet34. Loads weights from Models/plantDisease-resnet34.pth.
"""
import io
import os

# Class names must match the order used when the model was trained (from notebook dataset.classes).
CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy',
]

_model = None


def _get_model():
    """Lazy-load the model so the app can start even if torch is slow or missing."""
    global _model
    if _model is not None:
        return _model
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
        from PIL import Image
    except ImportError:
        raise RuntimeError(
            "torch, torchvision, and Pillow are required for prediction. "
            "Install with: pip install torch torchvision Pillow"
        )

    class PlantDiseaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            try:
                self.network = models.resnet34(weights=None)
            except TypeError:
                self.network = models.resnet34(pretrained=False)
            self.network.fc = nn.Linear(self.network.fc.in_features, 38)

        def forward(self, x):
            return self.network(x)

    _model = PlantDiseaseModel()
    _model.eval()

    # Path to .pth: same dir as this file -> Flask/ -> Models/plantDisease-resnet34.pth
    base = os.path.dirname(os.path.abspath(__file__))
    pth_path = os.path.join(base, "Models", "plantDisease-resnet34.pth")
    if not os.path.isfile(pth_path):
        raise FileNotFoundError(f"Model weights not found at {pth_path}")
    state = torch.load(pth_path, map_location="cpu", weights_only=True)
    _model.load_state_dict(state)
    return _model


def predict_image(img_bytes):
    """
    Run inference on image bytes (e.g. from request.files['file'].read()).
    Returns one of the CLASSES strings.
    """
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
        yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return CLASSES[preds[0].item()]
