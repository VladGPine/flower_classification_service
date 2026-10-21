"""ONNX model loading and prediction."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from app.errors import ModelLoadError
from app.preprocess import IMG_SIZE, preprocess

NUM_CLASSES = 104


def load_classes(path: Path) -> list[str]:
    try:
        classes = path.read_text(encoding='utf-8').splitlines()
    except OSError as err:
        raise ModelLoadError(f'cannot read classes file {path}') from err
    if len(classes) != NUM_CLASSES:
        raise ModelLoadError(f'{path}: expected {NUM_CLASSES} classes, got {len(classes)}')
    return classes


class FlowerClassifier:
    def __init__(self, model_path: Path, classes_path: Path):
        self.classes = load_classes(classes_path)
        try:
            self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        except Exception as err:
            raise ModelLoadError(f'cannot load ONNX model {model_path}') from err
        self.input_name = self.session.get_inputs()[0].name
        self._self_check()

    def _run(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: x})[0]

    def _self_check(self) -> None:
        """Run a dummy image: a model that loads but gives wrong output is also broken."""
        dummy = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        try:
            probs = self._run(dummy)
        except Exception as err:
            raise ModelLoadError(f'model failed on a {IMG_SIZE}x{IMG_SIZE} test input') from err
        if probs.shape != (1, NUM_CLASSES):
            raise ModelLoadError(f'unexpected model output shape {probs.shape}')
        if not np.isclose(probs.sum(), 1.0, atol=1e-3):
            raise ModelLoadError('model output is not a probability distribution')

    def predict(self, img: Image.Image) -> dict:
        probs = self._run(preprocess(img))[0]
        class_id = int(probs.argmax())
        return {
            'class_id': class_id,
            'class_name': self.classes[class_id],
            'probabilities': probs.tolist(),
        }