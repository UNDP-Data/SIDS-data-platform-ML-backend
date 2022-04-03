from enum import Enum


class Error(Enum):
    INVALID_TARGET_YEAR = {"msg": "Target year does not support. Support from {} to {}", "type": "value_error"}
    INVALID_DATASET = {"msg": "Invalid dataset. Supported types: {}", "type": "value_error"}
    INVALID_TARGET = {"msg": "Invalid target. Please try /imputation/targets endpoint for supported values",
                      "type": "value_error"}
    INVALID_PREDICTOR = {"msg": "Invalid predictor/s {}. Please try /imputation/predictors endpoint "
                                "for supported values", "type": "value_error"}

    def format(self, *argv):
        v = self
        v.value["msg"] = self.value["msg"].format(*argv)
        return v
