class InvalidImageError(Exception):
    """Uploaded file is not an acceptable image. The message is safe to show to the client."""


class ModelLoadError(Exception):
    """The ONNX model or the class list could not be loaded or failed the startup check."""