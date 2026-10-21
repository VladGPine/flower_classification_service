"""Image preprocessing — must match the notebook exactly."""

import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = 331
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img: Image.Image) -> np.ndarray:
    """PIL image -> float32 array (1, 3, IMG_SIZE, IMG_SIZE), normalized."""
    img = ImageOps.exif_transpose(img)
    img = img.convert('RGB')

    width, height = img.size
    if width < height:
        coef = IMG_SIZE / width
        result_width = IMG_SIZE
        result_height = round(coef * height)
    else:
        coef = IMG_SIZE / height
        result_height = IMG_SIZE
        result_width = round(coef * width)
    img = img.resize((result_width, result_height), Image.BILINEAR)

    left = (result_width - IMG_SIZE) // 2
    top = (result_height - IMG_SIZE) // 2
    img = img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))

    # to array, normalize, (H, W, C) -> (1, C, H, W)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    x = x.transpose(2, 0, 1)
    x = x[np.newaxis, ...]
    return x
