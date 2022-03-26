import sys

from fastapi import APIRouter

from common.definitions import TrainRequest, ModelResponse
from common.logger import logger
from models.imputation.model import query_and_train

import joblib

sys.modules['sklearn.externals.joblib'] = joblib

router = APIRouter(
    prefix="/imputation",
    tags=["Imputation"],
    responses={404: {"description": "Not found"}},
)
#
# @router.get('/params')
# async def imputation_params():
#     return


@router.post('/query_and_train')
async def train(req: TrainRequest):
    logger.info("Request received %s", req.target);
    avg_rmse, rmse, model_feature_importance, model_feature_names, prediction = query_and_train(req.manual_predictors, req.target_year, req.target, req.interpolator, req.scheme, req.estimators, req.model, req.interval, None)
    logger.info("Return values %f %f", rmse, avg_rmse);
    return ModelResponse(avg_rmse=avg_rmse, rmse=rmse, model_feature_importance=model_feature_importance, model_feature_names=model_feature_names, prediction=prediction)