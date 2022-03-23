from fastapi import APIRouter
from pydantic import BaseModel, create_model
from typing import List


router = APIRouter(
    prefix="/time_series",
    tags=["Time Series"],
    responses={404: {"description": "Not found"}},
)


@router.get('/params')
async def time_series_params():
    return


@router.post('/predict')
async def time_series_predict():
    return f"knn_predict"
