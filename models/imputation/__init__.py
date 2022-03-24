from fastapi import APIRouter

from common.definitions import TrainResponse, TrainRequest
from models.imputation.KNN_Based import query_and_train

router = APIRouter(
    prefix="/imputation",
    tags=["Imputation"],
    responses={404: {"description": "Not found"}},
)

@router.get('/params')
async def imputation_params():
    return


@router.post('/query_and_train')
async def train(req: TrainRequest):
    alert_msg, si_index, avp_fig, coef_fig, query_card, bar_fig_list_1, bar_fig_list_2, bar_fig_list_3 = query_and_train(req.manual_predictors, req.target_year, req.target, req.interpolator, req.scheme, req.estimators, req.model, req.interval, None)
    return TrainResponse(alert_msg=alert_msg, SI_index=si_index, avp_fig=avp_fig, coef_fig=coef_fig, query_card=query_card, bar_fig_list_1=bar_fig_list_1, bar_fig_list_2=bar_fig_list_2, bar_fig_list_3=bar_fig_list_3)