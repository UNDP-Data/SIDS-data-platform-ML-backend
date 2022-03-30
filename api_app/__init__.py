import fastapi
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware

app = fastapi.FastAPI(
    title="SIDS RestAPI",
    description="SIDS ML Backend RestAPI",
    version="0.1",
)

origins = [
    "https://lenseg.github.io/SIDSDataPlatform/",
    "https://sids-dashboard.github.io/SIDSDataPlatform/",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

