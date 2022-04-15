from enum import Enum

from pydantic import BaseModel, validator, root_validator
from pydantic.fields import ModelField


class BaseDefinition(BaseModel):
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
        schema = cls.schema()
        for key in schema["properties"]:
            val = schema["properties"][key]
            if "required_if" in val:
                matching = True
                for cond in val["required_if"]:
                    for c in cond:
                        if c in values:
                            if isinstance(values[c], Enum):
                                v = values[c].name
                            else:
                                v = str(values[c])

                            if v == cond[c] and (key not in values or values[key] is None):
                                matching = False
                                break
                if not matching:
                    raise ValueError(key + " field required for " + str(val["required_if"]))

        return values