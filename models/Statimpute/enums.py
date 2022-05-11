from enum import Enum



class method(Enum):
    linear = "linear",
    nearest = "nearest"
    zero = "zero",
    slinear = "slinear",
    quadratic ="quadratic",
    cubic = "cubic",
    spline = "spline",
    polynomial = "polynomial",
    piecewise_polynomial ="piecewise_polynomial",
    pchip = "pchip",
    akima ="akima",
    cubicspline = "cubicspline"