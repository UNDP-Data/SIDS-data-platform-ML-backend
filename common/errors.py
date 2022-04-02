from enum import Enum


class Error(Enum):
    TARGET_YEAR_NOT_EXIST = {"msg": "Target year does not exist in the dataset. Exist from %s to %s"}
