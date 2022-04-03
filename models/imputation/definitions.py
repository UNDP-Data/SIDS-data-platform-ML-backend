import enum
import logging
from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.fields import ModelField

from models.imputation import Interpolator, Schema, Model, Interval


class PredictorListRequest(BaseModel):
    target_year: str = Field(..., title="The year under consideration", example="2001")
    dataset: str = Field(..., title="Dataset", example="key")
    target: str = Field(..., title="Indicator whose values will be imputed", example="key-wdi-EG.ELC.ACCS.ZS")
    scheme: Schema = Field(..., title="Feature selection method selected by user", example=Schema.MANUAL.name)

    @validator('*', pre=True)
    def validate_all(cls, v, field: ModelField, config, values):
        if issubclass(field.type_, Enum):
            if v not in list(field.type_.__members__.keys()):
                raise ValueError("Invalid value for the field "+field.name+". Supported values:" + str(list(field.type_.__members__.keys())))
            else:
                v = field.type_[v]
        return v


class TrainRequest(BaseModel):
    manual_predictors: Optional[List[str]] = Field(None, title="List of predictors (for Manual)",
                                                   example=["wdi-AG.LND.AGRI.K2"])
    number_predictor: Optional[int] = Field(None, title="Number of predictors (for Automatic)", example=10)
    target_year: str = Field(..., title="The year under consideration", example="2001")
    target: str = Field(..., title="Indicator whose values will be imputed", example="key-wdi-EG.ELC.ACCS.ZS")
    interpolator: Interpolator = Field(..., title="Type of imputer to use for interpolation", example=Interpolator.KNNImputer.name)
    scheme: Schema = Field(..., title="Feature selection method selected by user", example=Schema.MANUAL.name)
    estimators: int = Field(..., title="Number of trees for tree based models", example=10)
    model: Model = Field(..., title="Type of model to be trained", example=Model.rfr.name)
    interval: Interval = Field(..., title="Type of prediction interval", example=Interval.quantile.name)
    dataset: str = Field(..., title="Dataset", example="key")

    @validator('*', pre=True)
    def validate_all(cls, v, field: ModelField, config, values):
        if issubclass(field.type_, Enum):
            if v not in list(field.type_.__members__.keys()):
                raise ValueError("Invalid value for the field "+field.name+". Supported values:" + str(list(field.type_.__members__.keys())))
            else:
                v = field.type_[v]
        return v

    @root_validator
    def validate_manual_predictors(cls, values):
        if "scheme" in values and values["scheme"] == Schema.MANUAL and ("manual_predictors" not in values or values["manual_predictors"] is None):
            raise ValueError("manual_predictors field required for MANUAL scheme")

        if "scheme" in values and values["scheme"] != Schema.MANUAL and (
                "number_predictor" not in values or values["number_predictor"] is None):
            raise ValueError("number_predictor field required for the given scheme")

        return values

class ModelResponse(BaseModel):
    rmse_deviation: Optional[float] = Field(..., description="Root-mean-square deviation")
    rmse: Optional[float] = Field(..., description="Root-mean-square deviation")
    model_feature_importance: Optional[List[float]]
    model_feature_names: Optional[List[str]]
    prediction: Optional[dict]
