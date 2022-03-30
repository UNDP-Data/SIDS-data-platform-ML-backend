import sys

from fastapi import APIRouter

from common.definitions import TrainRequest, ModelResponse
from common.logger import logger
from models.imputation.enums import Schema, Model, Interval, Interpolator
from models.imputation.model import query_and_train

router = APIRouter(
    prefix="/imputation",
    tags=["Imputation"],
    responses={404: {"description": "Not found"}},
)


@router.get('/params')
async def get_params():
    return {
        "Schema": Schema.__members__.items(),
        "Model": Model.__members__.items(),
        "Interpolator": Interpolator.__members__.items(),
        "Interval": Interval.__members__.items(),
    }


@router.post('/predict', response_model=ModelResponse)
async def train_validate_predict(req: TrainRequest):
    logger.info("Request received %s", req.target)

    manual_predictors = None
    if req.scheme == "Manual":
        manual_predictors = req.manual_predictors
    else:
        manual_predictors = req.number_predictor
    avg_rmse, rmse, model_feature_importance, model_feature_names, prediction = \
        query_and_train(manual_predictors, req.target_year,
                        req.target,
                        req.interpolator,
                        req.scheme,
                        req.estimators,
                        req.model,
                        req.interval, None)
    logger.info("Return values %f %f", rmse, avg_rmse)
    return ModelResponse(rmse_deviation=avg_rmse, rmse=rmse, model_feature_importance=model_feature_importance, model_feature_names=model_feature_names, prediction=prediction)
