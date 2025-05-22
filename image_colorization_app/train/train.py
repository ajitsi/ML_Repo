"""
Contains code for model training
"""
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

from app.model.colorization_model import ColorizationNet
from app.model.utils import to_grayscale

def train(model, dataloader, criterion, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for img, _ in dataloader:
            img = img.to(device)
            gray = to_grayscale(img)

            input_gray = gray
            target_ab = img[:, 1:] - input_gray.repeat(1, 2, 1, 1)  # Dummy AB channels

            optimizer.zero_grad()
            output = model(input_gray)
            loss = criterion(output, target_ab)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor()
    ])

    dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ColorizationNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trained_model = train(model, dataloader, criterion, optimizer, device, epochs=5)

    os.makedirs("app/model", exist_ok=True)
    torch.save(trained_model.state_dict(), "app/model/colorization_model.pth")
    print("✅ Model saved at app/model/colorization_model.pth")

if __name__ == "__main__":
    main()
