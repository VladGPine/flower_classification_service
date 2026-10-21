import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.errors import ModelLoadError
from app.main import app
from app.model import FlowerClassifier


def test_model_loads():
    clf = FlowerClassifier(settings.model_path, settings.classes_path)
    assert len(clf.classes) == 104


@pytest.mark.parametrize(
    'content',
    [b'', b'broken', b'\x00' * 1024],
    ids=['empty', 'text', 'zeros'],
)
def test_broken_onnx_is_rejected(tmp_path, content):
    broken = tmp_path / 'model.onnx'
    broken.write_bytes(content)

    with pytest.raises(ModelLoadError):
        FlowerClassifier(broken, settings.classes_path)


def test_wrong_classes_file_is_rejected(tmp_path):
    classes = tmp_path / 'classes.txt'
    classes.write_text('rose\ntulip\ndaisy\n', encoding='utf-8')
    with pytest.raises(ModelLoadError, match='104'):
        FlowerClassifier(settings.model_path, classes)


def test_app_does_not_start_with_broken_model(tmp_path, monkeypatch):
    broken = tmp_path / 'model.onnx'
    broken.write_bytes(b'broken')
    monkeypatch.setattr(settings, 'model_path', broken)

    # lifespan fails -> the client cannot start -> /health is never served
    with pytest.raises(ModelLoadError), TestClient(app):
        pass
