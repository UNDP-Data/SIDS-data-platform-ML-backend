from typing import Optional, List

from pydantic import BaseModel


class TrainRequest(BaseModel):
    manual_predictors: List[str]
    target_year: str
    target: str
    interpolator: str
    scheme: str
    estimators: int
    model: str
    interval: str
    ind_metadata: Optional[List]


class TrainResponse(BaseModel):
    alert_msg: str
    si_index: Optional[int]
    avp_fig: bytes
    coefficient_fig: Optional[str]
    query_card: Optional[str]
    bar_figure_1: Optional[str]
    bar_figure_2: Optional[str]
    bar_figure_3: Optional[str]


class ModelResponse(BaseModel):
    avg_rmse: Optional[float]
    rmse: Optional[float]
    model_feature_importance: Optional[List[int]]
    model_feature_names: Optional[List[str]]
    prediction: Optional[dict]
