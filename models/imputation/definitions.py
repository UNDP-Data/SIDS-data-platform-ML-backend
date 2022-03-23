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
    interval: int
    ind_metadata: Optional[List]


class TrainResponse(BaseModel):
    alert_msg: str
    si_index: int
    avp_fig: bytes
    coefficient_fig: bytes
    query_card: bytes
    bar_figure_1: bytes
    bar_figure_2: bytes
    bar_figure_3: bytes
