import io

from PIL import Image, UnidentifiedImageError

from app.config import settings
from app.errors import InvalidImageError

ALLOWED_FORMATS = {'JPEG', 'PNG'}


def decode_image(data: bytes) -> Image.Image:

    if not data:
        raise InvalidImageError('file is empty')
    if len(data) > settings.max_file_bytes:
        raise InvalidImageError('file is too large')

    try:
        img = Image.open(io.BytesIO(data))

        if img.format not in ALLOWED_FORMATS:
            raise InvalidImageError('only JPEG or PNG images are accepted')
        width, height = img.size
        if width * height > settings.max_pixels:
            raise InvalidImageError('image is too large')
        if min(width, height) < settings.min_side:
            raise InvalidImageError('image is too small')
        if max(width, height) / min(width, height) > settings.max_aspect_ratio:
            raise InvalidImageError('image aspect ratio is too extreme')

        img.load()
    except InvalidImageError:
        raise
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        SyntaxError,
        ValueError,
    ) as err:
        raise InvalidImageError('file is not a valid JPEG or PNG image') from err

    return img
