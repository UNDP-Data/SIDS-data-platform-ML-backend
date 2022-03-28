from typing import Optional, List

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    manual_predictors: Optional[List[str]] = Field(..., title="List of predictors (for Manual)",
                                                   example=["wdi-AG.LND.AGRI.K2"])
    number_predictor: Optional[int] = Field(..., title="Number of predictors (for Automatic)", example=10)
    target_year: str = Field(..., title="The year under consideration", example="2001")
    target: str = Field(..., title="Indicator whose values will be imputed", example="key-wdi-EG.ELC.ACCS.ZS")
    interpolator: str = Field(..., title="Type of imputer to use for interpolation", example="KNNImputer")
    scheme: str = Field(..., title="Feature selection method selected by user", example="Manual")
    estimators: int = Field(..., title="Number of trees for tree based models", example=10)
    model: str = Field(..., title="Type of model to be trained", example="rfr")
    interval: str = Field(..., title="Type of prediction interval", example="100")


class ModelResponse(BaseModel):
    rmse_deviation: Optional[float] = Field(..., description="Root-mean-square deviation")
    rmse: Optional[float] = Field(..., description="Root-mean-square deviation")
    model_feature_importance: Optional[List[int]]
    model_feature_names: Optional[List[str]]
    prediction: Optional[dict]
