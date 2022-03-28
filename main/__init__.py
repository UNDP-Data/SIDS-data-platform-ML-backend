import sys
import os
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import logging
import azure.functions as func
from api_app import app
from models import imputation

try:
    from azure.functions import AsgiMiddleware
except ImportError:
    from functions._http_asgi import AsgiMiddleware

app.include_router(imputation.router)


@app.get("/")
async def root():
    return {"message": "Hello SIDS ML Backend!"}


def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    return AsgiMiddleware(app).handle(req, context)
