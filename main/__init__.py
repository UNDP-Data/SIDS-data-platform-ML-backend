import sys
import os

from dependacy.azureFileHandler import AzureFileHandler

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
    return {"message": "Hello SIDS ML Backend V6!"}


@app.get("/datasets")
async def list_datasets():
    return AzureFileHandler().list_files("datasets")


def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    return AsgiMiddleware(app).handle(req, context)
