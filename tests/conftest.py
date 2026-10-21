import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import settings
from app.main import app


@pytest.fixture(scope='session')
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope='session')
def classes() -> list[str]:
    return settings.classes_path.read_text(encoding='utf-8').splitlines()


@pytest.fixture
def make_image():

    def _make(
        fmt: str = 'JPEG', size: tuple[int, int] = (400, 300), mode: str = 'RGB'
    ) -> bytes:
        buf = io.BytesIO()
        Image.new(mode, size, color=128).save(buf, format=fmt)
        return buf.getvalue()

    return _make
