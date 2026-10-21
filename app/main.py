"""FastAPI service: flower species by photo."""
 
from contextlib import asynccontextmanager
 
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import router
from app.config import settings
from app.errors import InvalidImageError
from app.model import FlowerClassifier
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Any error here (missing/broken/incompatible ONNX, bad classes.txt)
    # aborts startup: the process exits and /health never answers 200.
    app.state.classifier = FlowerClassifier(settings.model_path, settings.classes_path)
    yield
 
 
app = FastAPI(title='Flower classifier', lifespan=lifespan)
 
 
@app.exception_handler(InvalidImageError)
async def invalid_image_handler(request: Request, exc: InvalidImageError) -> JSONResponse:
    return JSONResponse(status_code=400, content={'detail': str(exc)})


app.include_router(router)