from enum import Enum


class Interval(Enum):
    bootstrap='Residual Bootstrap'
    quantile='Quantile'

class Schema(Enum):
    AUTO="Automatic"
    MANUAL="Manual"


class Model(Enum):
    rfr='Random Forest Regressor'
    gbr='Gradient Boost Regressor'
    etr='Extra tree Regressor'

    all='Best Model'

