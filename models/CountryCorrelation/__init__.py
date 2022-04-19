from typing import Optional

from fastapi import APIRouter
from pydantic import Field

from common.base_definition import BaseDefinition
from models.CountryCorrelation.correlation import correlation_function,cluster_function


class CorrRequest(BaseDefinition):
    dataset: str = Field(..., title="The dataset code as defined in indicatorMeta file", example="wdi")
    category: str = Field(..., title="The category of the indicator as defined in indicatorMeta file ", example="Financial Sector")
    country: str = Field(..., title="the iso-3 code for the SIDS country", example="SGP")
    year: int = Field(..., title = "The year under consideration", example= 2014)

class ClusRequest(BaseDefinition):
    dataset: str = Field(..., title="The dataset code as defined in indicatorMeta file", example="wdi")
    category: str = Field(..., title="The category of the indicator as defined in indicatorMeta file ", example="Financial Sector")
    year: int = Field(..., title = "The year under consideration", example= 2014)
    k: int = Field(..., title = "The number of clusters expected", example= 5)


class CorrResponse(BaseDefinition):
    country_corr: Optional[dict]

class ClusResponse(BaseDefinition):
    country: Optional[list]
    label: Optional[list]




router = APIRouter(
    prefix="/correlation",
    tags=["Country Correlation Model"],
    responses={404: {"description": "Not found"}},
)


@router.post('/correlate', response_model=CorrResponse)
async def correlation(req: CorrRequest):
    country_corr= correlation_function(req.dataset,req.category,req.country,req.year) 
    return CorrResponse(country_corr=country_corr)

@router.post('/cluster', response_model=ClusResponse)
async def clustering(req: ClusRequest):
    country,label= cluster_function(req.dataset,req.category,req.year,req.k) 
    return ClusResponse(country=country,label=label)


