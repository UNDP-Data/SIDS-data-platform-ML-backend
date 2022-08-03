import json
import time

from fastapi import APIRouter, HTTPException
from fastapi.params import Depends

from common.constants import MAIN_ENDPOINT_TAG
from common.errors import Error
from models.timeseriesImp.enums import Model, Interval
from common.logger import logger

from typing import Optional, List

from pydantic import BaseModel, Field

from common.base_definition import BaseDefinition

from models.timeseriesImp.model import  train_predict,predictor_validity,target_validity,year_validity

router = APIRouter(
    prefix="/timeseriesImp",
    tags=["Timeseries Imputation"],
    responses={404: {"description": "Not found"}},
    
)

class TrainRequest(BaseDefinition):
    manual_predictors: Optional[List[str]] = Field(None, title="List of predictors",
                                                   example=[
                                                            'wdi2-SE.SEC.ENRL.FE.ZS',
                                                            'wdi2-NY.GDP.MINR.RT.ZS',
                                                            'wdi2-SE.SEC.ENRL.VO.FE.ZS',
                                                            'wdi2-TM.VAL.MRCH.R4.ZS',
                                                            'wdi2-TX.VAL.MRCH.R6.ZS',
                                                            'wdi2-NY.GNP.PCAP.CN',
                                                            'wdi2-SP.DYN.CBRT.IN',
                                                            'wdi2-NY.GDP.PCAP.CD',
                                                            'wdi-ER.FSH.CAPT.MT'], req_endpoint="/predictors")
    #number_predictor: Optional[int] = Field(None, title="Number of predictors (for Automatic)", example=10,
    #                                        required_if=[{"scheme": "AFS"}, {"scheme": "PCA"}])
    target_year: int = Field(..., title="The year under consideration", example=2010,req_endpoint="/target_years")
    target: str = Field(..., title="Indicator whose values will be imputed", example="wdi2-SE.PRM.ENRL.TC.ZS",
                        req_endpoint="/targets")
    estimators: int = Field(..., title="Number of trees for tree based models", example=10)
    model: Model = Field(..., title="Type of model to be trained", example=Model.rfr.name)
    interval: Interval = Field(..., title="Type of prediction interval", example=Interval.quantile.name)
    #dataset: str = Field(..., title="Dataset", example="key", req_endpoint="/datasets")


class ModelResponse(BaseDefinition):
    #rmse_deviation: Optional[float] = Field(..., description="Root-mean-square deviation")
    rmse: Optional[float] = Field(..., description="Root-mean-square deviation")
    model_feature_importance: Optional[List[float]]
    model_feature_names: Optional[List[str]]
    prediction: Optional[dict]
    #correlation: Optional[dict]
    #feature_importance_pie: Optional[dict]

@router.get('/targets')
async def get_targets():

    return target_validity()

@router.get('/predictors')
async def get_predictors(target: str):

    return predictor_validity(target)

@router.get('/target_years')
async def get_years(target: str):
    return year_validity(target)



@router.post('/predict', response_model=ModelResponse, openapi_extra={MAIN_ENDPOINT_TAG: True})
async def train_validate_predict(req: TrainRequest):
    #received_time = int(time.time())
    #check_year_validity(req.target_year)
    #check_dataset_validity(req.target_year, req.dataset)
    #check_target_validity(req.target_year, req.dataset, req.target)

    logger.info("Request received %s", req.target)

    prediction,rmse, model_feature_importance, model_feature_names = \
        train_predict(req.manual_predictors,req.estimators,req.model,req.target,req.target_year,req.interval)

    #logger.info("Return values %f %f", rmse, avg_rmse)
    resp = ModelResponse(rmse=rmse, model_feature_importance=model_feature_importance,
                         model_feature_names=model_feature_names, prediction=prediction)
    #time_consumed = int(time.time()) - received_time
    #logger.info("Time Consumed(s)=%d %s", time_consumed, str(req))
    return resp