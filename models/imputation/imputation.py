from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel


class DatasetParams(BaseModel):
    name: str


class YearParams(BaseModel):
    year: int
    datasets: List[DatasetParams]


class ModelParams(BaseModel):
    years: List[YearParams]


router = APIRouter(
    prefix="/imputation",
    tags=["Imputation"],
    responses={404: {"description": "Not found"}},
)


@router.get('/params', response_model=ModelParams)
async def imputation_params():
    return


@router.post('/predict')
async def imputation_predict():
    return f"knn_predict"