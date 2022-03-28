import fastapi
from fastapi.openapi.utils import get_openapi

app = fastapi.FastAPI(
    title="SIDS RestAPI",
    description="SIDS ML Backend RestAPI",
    version="0.1",
)