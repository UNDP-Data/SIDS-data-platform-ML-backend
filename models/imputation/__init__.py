from fastapi import APIRouter

from definitions import TrainRequest, TrainResponse
from models.imputation.KNN_Based import query_and_train
from models.imputation.utils import preprocessing, data_importer

router = APIRouter(
    prefix="/imputation",
    tags=["Imputation"],
    responses={404: {"description": "Not found"}},
)

wb_data,indicatorMeta, datasetMeta,indicatorData = data_importer(model_type="knn")

@router.get('/params')
async def imputation_params():
    return


@router.post('/query_and_train', response_model=TrainResponse)
async def train(req: TrainRequest):
    alert_msg, si_index, avp_fig, coef_fig, query_card, bar_fig_list_1, bar_fig_list_2, bar_fig_list_3 = query_and_train(req.manual_predictors, req.target_year, req.target, req.interpolator, req.scheme, req.estimators, req.model, req.interval)
    return TrainResponse(alert_msg=alert_msg, SI_index=si_index, avp_fig=avp_fig, coef_fig=coef_fig, query_card=query_card, bar_fig_list_1=bar_fig_list_1, bar_fig_list_2=bar_fig_list_2, bar_fig_list_3=bar_fig_list_3)