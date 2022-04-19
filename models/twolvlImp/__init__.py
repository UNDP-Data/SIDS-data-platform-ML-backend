import json
import time

from fastapi import APIRouter, HTTPException
from fastapi.params import Depends

from common.errors import Error
from models.twolvlImp.enums import Schema, Model, Interval, Interpolator
from models.twolvlImp.definitions import TrainRequest, ModelResponse, PredictorListRequest
from common.logger import logger
from models.twolvlImp.model import query_and_train, get_indicator_list, get_predictor_list, get_target_years, \
    dataset_options, dimension_options, check_year_validity, check_dataset_validity, check_target_validity, check_predictors_validity, \
    target_sample_size_requirement, predictor_sample_size_requirement

router = APIRouter(
    prefix="/twolvlImp",
    tags=["Two Level Imputation"],
    responses={404: {"description": "Not found"}},
)


# @router.get('/params')
# async def get_params():
#     return {
#         "Schema": Schema.__members__.items(),
#         "Model": Model.__members__.items(),
#         "Interpolator": Interpolator.__members__.items(),
#         "Interval": Interval.__members__.items(),
#     }

@router.get('/target_sample_size')
async def get_target_sample_size():
    return target_sample_size_requirement()


@router.get('/predictor_sample_size')
async def get_predictor_sample_size():
    return predictor_sample_size_requirement()


@router.get('/target_years')
async def get_years():
    return get_target_years()


@router.get('/datasets')
async def get_datasets(target_year: str = "2001"):
    check_year_validity(target_year)
    return dataset_options(target_year)


@router.get('/targets')
async def get_indicators(target_year: str, dataset: str):
    check_year_validity(target_year)
    check_dataset_validity(target_year, dataset)

    return get_indicator_list(target_year, dataset)


@router.get('/dimensions')
async def get_dimensions(target: str, target_year: str, dataset: str):
    check_year_validity(target_year)
    check_dataset_validity(target_year, dataset)
    check_target_validity(target_year, dataset, target)

    return dimension_options(target, target_year)


@router.post('/predictors')
async def get_predictors(args: PredictorListRequest):

    check_year_validity(args.target_year)
    check_dataset_validity(args.target_year, args.dataset)
    check_target_validity(args.target_year, args.dataset, args.target)
    return get_predictor_list(args.target, args.target_year, args.scheme)


@router.post('/estimate')
async def estimate_time(req: TrainRequest):
    check_year_validity(req.target_year)
    check_dataset_validity(req.target_year, req.dataset)
    check_target_validity(req.target_year, req.dataset, req.target)
    if req.scheme == Schema.MANUAL:
        return 30
    else:
        return 2*60


@router.post('/predict', response_model=ModelResponse)
async def train_validate_predict(req: TrainRequest):
    received_time = int(time.time())
    check_year_validity(req.target_year)
    check_dataset_validity(req.target_year, req.dataset)
    check_target_validity(req.target_year, req.dataset, req.target)

    logger.info("Request received %s", req.target)


    if req.scheme == Schema.MANUAL:
        manual_predictors = req.manual_predictors
        check_predictors_validity(req.target_year, req.manual_predictors)

    else:
        manual_predictors = req.number_predictor

    avg_rmse, rmse, model_feature_importance, model_feature_names, prediction,correlation,feature_importance_pie = \
        query_and_train(manual_predictors, req.target_year,
                        req.target,
                        req.interpolator,
                        req.scheme,
                        req.estimators,
                        req.model,
                        req.interval, None)
    logger.info("Return values %f %f", rmse, avg_rmse)
    resp = ModelResponse(rmse_deviation=avg_rmse, rmse=rmse, model_feature_importance=model_feature_importance, model_feature_names=model_feature_names, prediction=prediction,correlation=correlation,feature_importance_pie=feature_importance_pie)
    time_consumed = int(time.time()) - received_time
    logger.info("service=%s time_consumed(s)=%d %s", "twolvlImp", time_consumed, str(req))
    return resp
