

import torch

def to_grayscale(tensor_img):
    # Convert RGB image to grayscale using average method
    r, g, b = tensor_img[:, 0:1, :, :], tensor_img[:, 1:2, :, :], tensor_img[:, 2:3, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
