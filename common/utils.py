import inspect
import sys
from enum import Enum

from fastapi import APIRouter
from fastapi.routing import APIRoute

from common.constants import MAIN_ENDPOINT_TAG


def save_file(filename, data):
    with open(filename, 'wb') as f:
        f.write(data)


def get_schema_inputs(route, enum_map, route_map):
    inputs = []
    if route.body_field is not None:
        schema = route.body_field.type_.schema()
        for key in schema["properties"]:
            val = schema["properties"][key]

            par = {"parameter": key,
                   "label": val["title"]}
            if "description" in val:
                par['description'] = val["description"]
            if "type" in val:
                par['type'] = val["type"]

            elif "allOf" in val and len(val["allOf"]) > 0:
                enm = val["allOf"][0]["$ref"].replace("#/definitions/", "")
                if enm in enum_map:
                    enm_def = enum_map[enm]
                    par['type'] = "enum"
                    par["values"] = enm_def.__members__.items()

            if "req_endpoint" in val:
                par["req_endpoint"] = {"endpoint": val["req_endpoint"]}
                if val["req_endpoint"] in route_map:
                    rout = route_map[val["req_endpoint"]]
                    for e in rout.methods:
                        par["req_endpoint"]["method"] = e
                        break
                    par["req_endpoint"]["inputs"] = get_schema_inputs(rout, enum_map, route_map)

            if "required" in schema and key in schema["required"]:
                par["conditions"] = [
                    {"required": True}
                ]
            elif "required_if" in val:
                par["conditions"] = [
                    {"required": val["required_if"]}
                ]

            inputs.append(par)
    elif route.dependant is not None and route.dependant.query_params is not None:
        for q in route.dependant.query_params:
            print(q)
            par = {"parameter": q.name,
                   "type": q.type_.__name__}
            if q.required:
                par["conditions"] = [
                    {"required": True}
                ]
            inputs.append(par)
        # inputs.append("")
    return inputs


def get_param_obj(model_name, router: APIRouter):

    resp = {
        "model": model_name,
        "label": router.tags[0] if len(router.tags) > 0 else None,
    }

    route_map = {}
    r: APIRoute
    main_route = None
    for r in router.routes:
        route_map[r.path.replace("/" + model_name, "")] = r
        if r.openapi_extra is not None and MAIN_ENDPOINT_TAG in r.openapi_extra and r.openapi_extra[MAIN_ENDPOINT_TAG]:
            main_route = r

    # TODO: Curretly assuming there will be only one main endpoint in one model.
    #  If this behvior needs to change, update the params endpoints data structure.

    if main_route is not None:
        enum_map = {}
        for name, obj in inspect.getmembers(sys.modules["models." + model_name]):
            if inspect.isclass(obj) and issubclass(obj, Enum):
                enum_map[obj.__name__] = obj

        if main_route.body_field is not None:
            resp["inputs"] = get_schema_inputs(main_route, enum_map, route_map)

    return resp
