from typing import Literal

from pydantic import BaseModel


class PredictResponse(BaseModel):
    class_id: int
    class_name: str
    probabilities: list[float]


class HealthResponse(BaseModel):
    status: Literal['ok']


class ErrorResponse(BaseModel):
    detail: str
