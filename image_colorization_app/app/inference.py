# app/inference.py
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from app.model.colorization_model import ColorizationNet

def load_model(checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(os.path.dirname(__file__), "model", "colorization_model.pth")
    model = ColorizationNet()
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    return model

def preprocess(image):
    image = image.convert("L").resize((224, 224))
    transform = transforms.ToTensor()
    L = transform(image).unsqueeze(0)
    return L

def postprocess(output_ab, input_L):
    ab = output_ab.detach().numpy().squeeze().transpose(1, 2, 0)
    # Ensure input_L has shape (1, 224, 224)
    L = input_L.squeeze().numpy()
    if L.ndim == 2:
        # Shape is (224, 224), add channel axis
        L = L[np.newaxis, ...]
    L = np.transpose(L, (1, 2, 0))  # (224, 224, 1)
    LAB = np.concatenate((L, ab), axis=2)
    LAB = (LAB * 255).astype("uint8")
    return Image.fromarray(LAB, mode="LAB").convert("RGB")

def colorize_image(image, model):
    input_L = preprocess(image)
    output_ab = model(input_L)
    return postprocess(output_ab, input_L)
