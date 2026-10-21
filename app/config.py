from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_path: Path = Path('model/model.onnx')
    classes_path: Path = Path('model/classes.txt')
    max_file_bytes: int = 20 * 1024 * 1024
    max_pixels: int = 50_000_000
    min_side: int = 32
    max_aspect_ratio: float = 10


settings = Settings()
