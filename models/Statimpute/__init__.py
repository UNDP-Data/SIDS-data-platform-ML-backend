
from fastapi import APIRouter, HTTPException
from fastapi.params import Depends

from common.constants import MAIN_ENDPOINT_TAG
from common.errors import Error

from common.logger import logger
from common.base_definition import BaseDefinition
from typing import Optional, List
from fastapi import APIRouter
from pydantic import Field
from models.Statimpute.model import imputer,check_country_validity,get_country_list


class statRequest(BaseDefinition):

    target: str = Field(..., title="Indicator whose values will be imputed", example="wdi-EG.ELC.ACCS.ZS")
    country: str= Field(..., title="IS0 3 code of the SIDS country to be imputed", example="CUB")
    method: str = Field(..., title="method for timeseries interpolation", example='linear')
    direction: str = Field(..., title="method for timeseries interpolation in time (forward, backward or both)", example='both') #maybe make optional
    d: int = Field(None, title="order of the polynomials (1,2 or 3)",
                                                   example=1, req_endpoint="/impute",
                                                   required_if=[{"method": "spline"},{"method":"polynomial"}])
class statResponse(BaseDefinition):
    missing_years: Optional[List[str]] = Field(..., description="list with the years that were imputed")
    imputed_series: Optional[dict] = Field(..., description="dict with the series of years and values")


router = APIRouter(
    prefix="/statimpute",
    tags=["Functional Imputation"],
    responses={404: {"description": "Not found"}},
)

@router.get('/countries')
async def get_countries(target: str):
    return get_country_list(target)

@router.post('/impute', response_model=statResponse, openapi_extra={MAIN_ENDPOINT_TAG: True})
async def predict(req: statRequest):
    check_country_validity(req.target,req.country)
    try:
        direction = req.direction
    except:
        direction = 'both'

    try:
        d = req.d
    except:
        d=1
    missing_years,imp_data = imputer(req.target, req.country,req.method,direction=direction,d=d)
    resp = statResponse(missing_years=missing_years,imputed_series=imp_data)
    return resp