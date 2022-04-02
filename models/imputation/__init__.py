from fastapi import APIRouter
from fastapi.params import Depends

from models.imputation.enums import Schema, Model, Interval, Interpolator
from models.imputation.definitions import TrainRequest, ModelResponse, PredictorListRequest
from common.logger import logger
from models.imputation.model import query_and_train, get_indicator_list, get_predictor_list, get_target_years, \
    dataset_options, dimension_options

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


@router.get('/target_years')
async def get_years():
    return get_target_years()


@router.get('/datasets')
async def get_datasets(target_year: str = "2001"):
    return dataset_options(target_year)


@router.get('/targets')
async def get_indicators(target_year: str, dataset: str):
    return get_indicator_list(target_year, dataset)


@router.post('/predictors')
async def get_predictors(args: PredictorListRequest):
    return get_predictor_list(args.target, args.target_year, args.scheme)


@router.get('/dimensions')
async def get_dimensions(target: str, target_year: str):
    return dimension_options(target, target_year)


@router.post('/predict', response_model=ModelResponse)
async def train_validate_predict(req: TrainRequest):
    logger.info("Request received %s", req.target)

    if req.scheme == Schema.MANUAL:
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
