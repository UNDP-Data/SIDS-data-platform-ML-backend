from typing import Optional, List

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    manual_predictors: Optional[List[str]] = ["wdi-AG.LND.AGRI.K2"]
    number_predictor: Optional[int] = 10
    target_year: str = "2001"
    target: str = "key-wdi-EG.ELC.ACCS.ZS"
    interpolator: str = "KNNImputer"
    scheme: str = "Manual"
    estimators: int = 10
    model: str = "rfr"
    interval: str = "100"


class ModelResponse(BaseModel):
    rmse_deviation: Optional[float] = Field(..., description="Root-mean-square deviation")
    rmse: Optional[float] = Field(..., description="Root-mean-square deviation")
    model_feature_importance: Optional[List[int]]
    model_feature_names: Optional[List[str]]
    prediction: Optional[dict]
