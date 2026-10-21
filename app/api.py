from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from app.config import settings
from app.images import decode_image
from app.model import FlowerClassifier
from app.schemas import ErrorResponse, HealthResponse, PredictResponse

router = APIRouter()


def get_classifier(request: Request) -> FlowerClassifier:
    return request.app.state.classifier


def bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


@router.get('/health', response_model=HealthResponse)
def health() -> dict:
    return {'status': 'ok'}


@router.post(
    '/predict',
    response_model=PredictResponse,
    responses={400: {'model': ErrorResponse}},
)
async def predict(
    request: Request, classifier: Annotated[FlowerClassifier, Depends(get_classifier)]
) -> dict:
    try:
        async with request.form() as form:
            uploads = [v for _, v in form.multi_items() if isinstance(v, UploadFile)]
            if len(uploads) > 1:
                raise bad_request('exactly one image is allowed')
            upload = form.get('file')
            if not isinstance(upload, UploadFile):
                raise bad_request('field `file` with an image is required')
            data = await upload.read(settings.max_file_bytes + 1)
    except HTTPException:
        raise
    except Exception as err:
        raise bad_request(
            'request must be multipart/form-data with a `file` field'
        ) from err

    img = decode_image(data)

    return await run_in_threadpool(classifier.predict, img)
