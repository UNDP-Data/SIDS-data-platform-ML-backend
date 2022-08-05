from enum import Enum


class Error(Enum):
    INVALID_TARGET_YEAR = {"msg": "Target year does not support. Support from {} to {}", "type": "value_error"}
    INVALID_DATASET = {"msg": "Invalid dataset. Supported types: {}", "type": "value_error"}
    INVALID_TARGET = {"msg": "Invalid target. Please try targets endpoint in the same path for supported values",
                      "type": "value_error"}
    INVALID_PREDICTOR = {"msg": "Invalid predictor/s {}. Please try /predictors endpoint in the same path "
                                "for supported values", "type": "value_error"}
    INVALID_COUNTRY= {"msg": "Invalid country {}. Please try /statimpute/get_countries endpoint "
                            "for supported values", "type": "value_error"}
    NONEXISTENT_TARGET = {"msg": "The indicator Code does not exist"}
    def format(self, *argv):
        v = self
        v.value["msg"] = self.value["msg"].format(*argv)
        return v
