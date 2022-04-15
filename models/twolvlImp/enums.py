from enum import Enum


class Schema(Enum):
    AFS="Automatic via feature selection"
    PCA="Automatic via PCA"
    MANUAL="Manual"


class Interpolator(Enum):
    KNNImputer='K Nearest Neighbour Interpolation'
    IterativeImputer='Iterative Imputer Interpolation'
    SimpleImputer='Simple Interpolation'


class Interval(Enum):
    bootstrap='Residual Bootstrap'
    quantile='Quantile'


class Model(Enum):
    rfr='Random Forest Regressor'
    gbr='Gradient Boost Regressor'
    etr='Extra tree Regressor'
    all='Best Model'
