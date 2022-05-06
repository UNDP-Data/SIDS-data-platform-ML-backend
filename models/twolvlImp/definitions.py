from typing import Optional, List

from pydantic import BaseModel, Field

from common.base_definition import BaseDefinition
from models.twolvlImp import Interpolator, Schema, Model, Interval


class PredictorListRequest(BaseDefinition):
    target_year: str = Field(..., title="The year under consideration", example="2001")
    dataset: str = Field(..., title="Dataset", example="key")
    target: str = Field(..., title="Indicator whose values will be imputed", example="key-wdi-EG.ELC.ACCS.ZS")
    scheme: Schema = Field(..., title="Feature selection method selected by user", example=Schema.MANUAL.name)


class TrainRequest(BaseDefinition):
    manual_predictors: Optional[List[str]] = Field(None, title="List of predictors (for Manual)",
                                                   example=["wdi-AG.LND.AGRI.K2"], req_endpoint="/predictors",
                                                   required_if=[{"scheme": "MANUAL"}])
    number_predictor: Optional[int] = Field(None, title="Number of predictors (for Automatic)", example=10,
                                            required_if=[{"scheme": "AFS"}, {"scheme": "PCA"}])
    target_year: str = Field(..., title="The year under consideration", example="2001", req_endpoint="/target_years")
    target: str = Field(..., title="Indicator whose values will be imputed", example="key-wdi-EG.ELC.ACCS.ZS",
                        req_endpoint="/targets")
    interpolator: Interpolator = Field(..., title="Base Imputer", description="The base imputer is a standard model "
                                                                              "that will be used to interpolate missing"
                                                                              " values in the most complete prediction "
                                                                              "features", example=Interpolator.KNNImputer.name)
    scheme: Schema = Field(..., title="Feature selection method selected by user", example=Schema.MANUAL.name)
    estimators: int = Field(..., title="Number of trees for tree based models", example=10)
    model: Model = Field(..., title="Type of model to be trained", example=Model.rfr.name)
    interval: Interval = Field(..., title="Type of prediction interval", example=Interval.quantile.name)
    dataset: str = Field(..., title="Dataset", example="key", req_endpoint="/datasets")


class ModelResponse(BaseDefinition):
    sample_size:Optional[float] = Field(..., description="number of samples used for training")
    rmse_deviation: Optional[float] = Field(..., description="Root-mean-square deviation")
    rmse: Optional[float] = Field(..., description="Root-mean-square deviation")
    model_feature_importance: Optional[List[float]]
    model_feature_names: Optional[List[str]]
    prediction: Optional[dict]
    correlation: Optional[dict]
    feature_importance_pie: Optional[dict]
