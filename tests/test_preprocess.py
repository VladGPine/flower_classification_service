import io

import numpy as np
import pytest
from PIL import Image

from app.preprocess import IMG_SIZE, MEAN, STD, preprocess


@pytest.mark.parametrize(
    'size',
    [(IMG_SIZE, IMG_SIZE), (640, 480), (300, 900), (50, 60), (2000, 1500)],
    ids=['model-size', 'landscape', 'portrait', 'smaller', 'larger'],
)
@pytest.mark.parametrize('mode', ['RGB', 'RGBA', 'L', 'CMYK'])
def test_output_shape_and_dtype(size, mode):
    x = preprocess(Image.new(mode, size))
    assert x.shape == (1, 3, IMG_SIZE, IMG_SIZE)
    assert x.dtype == np.float32


def test_normalization():
    # a uniform image with the ImageNet mean color must become ~0 everywhere
    mean_rgb = tuple(round(float(m) * 255) for m in MEAN)
    x = preprocess(Image.new('RGB', (IMG_SIZE, IMG_SIZE), mean_rgb))
    assert np.abs(x).max() < 1 / 255 / STD.min()


def test_center_crop_keeps_center():
    # portrait image: red top quarter, green middle, blue bottom quarter
    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE * 2), (0, 255, 0))
    img.paste((255, 0, 0), (0, 0, IMG_SIZE, IMG_SIZE // 2))
    img.paste((0, 0, 255), (0, IMG_SIZE * 3 // 2, IMG_SIZE, IMG_SIZE * 2))

    x = preprocess(img)[0]
    pixel_rgb = x[:, IMG_SIZE // 2, IMG_SIZE // 2] * STD + MEAN
    assert pixel_rgb == pytest.approx([0.0, 1.0, 0.0], abs=0.01)


def test_exif_orientation_is_applied():
    # landscape pixels + EXIF "rotate 90" -> must be treated as portrait;
    # left half white, right half black before rotation
    img = Image.new('RGB', (600, 400), (0, 0, 0))
    img.paste((255, 255, 255), (0, 0, 300, 400))
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90 CW
    rotated = Image.open(_save_jpeg_with_exif(img, exif))

    x = preprocess(rotated)[0]
    top = x[:, 5, IMG_SIZE // 2] * STD + MEAN
    bottom = x[:, IMG_SIZE - 5, IMG_SIZE // 2] * STD + MEAN
    # after rotating 90 CW the white (left) half ends up on top
    assert top.mean() > 0.9
    assert bottom.mean() < 0.1


def _save_jpeg_with_exif(img: Image.Image, exif: Image.Exif) -> io.BytesIO:
    buf = io.BytesIO()
    img.save(buf, format='JPEG', exif=exif, quality=95)
    buf.seek(0)
    return buf
