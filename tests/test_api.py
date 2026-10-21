from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / 'data'
MIME = {'JPEG': 'image/jpeg', 'PNG': 'image/png', 'GIF': 'image/gif'}


def post_image(
    client, data: bytes, filename: str = 'image.jpg', content_type: str = 'image/jpeg'
):
    return client.post('/predict', files={'file': (filename, data, content_type)})


def assert_bad_request(response):
    assert response.status_code == 400, f'actual status code: {response.status_code}'
    detail = response.json()['detail']
    assert isinstance(detail, str) and detail


def test_health(client):
    response = client.get('/health')
    assert response.status_code == 200, f'actual status code: {response.status_code}'
    assert response.json() == {'status': 'ok'}


@pytest.mark.parametrize('fmt', ['JPEG', 'PNG'])
@pytest.mark.parametrize(
    'size',
    [(331, 331), (640, 480), (300, 900)],
    ids=['square', 'landscape', 'portrait'],
)
def test_predict_valid_image(client, classes, make_image, fmt, size):
    response = post_image(client, make_image(fmt, size), content_type=MIME[fmt])
    assert response.status_code == 200, f'actual status code: {response.status_code}'

    body = response.json()
    assert set(body) == {'class_id', 'class_name', 'probabilities'}, (
        f'actual body: {set(body)}'
    )

    class_id = body['class_id']
    probs = body['probabilities']
    assert isinstance(class_id, int) and 0 <= class_id <= 103, (
        f'actual class id: {class_id}'
    )
    assert body['class_name'] == classes[class_id], (
        f'actual class name: {body["class_name"]}'
    )
    assert len(probs) == 104, f'actual probabilities length: {len(probs)}'
    assert sum(probs) == pytest.approx(1.0, abs=1e-3), (
        f'actual sum of probabilities: {sum(probs)}'
    )
    assert max(range(104), key=probs.__getitem__) == class_id


def test_predict_png_with_alpha(client, make_image):
    response = post_image(
        client, make_image('PNG', mode='RGBA'), content_type='image/png'
    )
    assert response.status_code == 200, f'actual status code: {response.status_code}'


# Real flower photos: put them in tests/data/ named "<class_id>_<anything>.jpg",
# e.g. "57_geranium.jpg". The test checks the predicted class.
FLOWER_PHOTOS = sorted(DATA_DIR.glob('*_*.jp*g')) + sorted(DATA_DIR.glob('*_*.png'))


@pytest.mark.skipif(not FLOWER_PHOTOS, reason='no photos in tests/data')
@pytest.mark.parametrize('photo', FLOWER_PHOTOS, ids=lambda p: p.name)
def test_predict_real_flower(client, photo):
    expected = int(photo.name.split('_')[0])
    response = post_image(client, photo.read_bytes(), filename=photo.name)
    assert response.status_code == 200, f'actual status code: {response.status_code}'
    assert response.json()['class_id'] == expected, (
        f'actual class id: {response.json()["class_id"]}'
    )


def test_empty_file(client):
    assert_bad_request(post_image(client, b''))


def test_text_file(client):
    assert_bad_request(
        post_image(client, b'hello, I am not an image', filename='a.jpg')
    )


def test_gif(client, make_image):
    assert_bad_request(
        post_image(
            client, make_image('GIF'), filename='a.gif', content_type='image/gif'
        )
    )


def test_truncated_jpeg(client, make_image):
    data = make_image('JPEG', (640, 480))
    assert_bad_request(post_image(client, data[: len(data) // 2]))


def test_too_small_image(client, make_image):
    assert_bad_request(post_image(client, make_image('JPEG', (10, 10))))


def test_extreme_aspect_ratio(client, make_image):
    assert_bad_request(post_image(client, make_image('JPEG', (800, 40))))


def test_two_files(client, make_image):
    data = make_image()
    files = [
        ('file', ('a.jpg', data, 'image/jpeg')),
        ('file', ('b.jpg', data, 'image/jpeg')),
    ]
    assert_bad_request(client.post('/predict', files=files))


def test_two_files_in_different_fields(client, make_image):
    data = make_image()
    files = {
        'file': ('a.jpg', data, 'image/jpeg'),
        'other': ('b.jpg', data, 'image/jpeg'),
    }
    assert_bad_request(client.post('/predict', files=files))


def test_wrong_field_name(client, make_image):
    response = client.post(
        '/predict', files={'image': ('a.jpg', make_image(), 'image/jpeg')}
    )
    assert_bad_request(response)


def test_no_body(client):
    assert_bad_request(client.post('/predict'))


def test_json_body(client):
    assert_bad_request(client.post('/predict', json={'file': 'flower.jpg'}))
