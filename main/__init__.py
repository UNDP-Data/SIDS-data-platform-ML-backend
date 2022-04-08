import importlib
import sys
import os
from typing import List

from fastapi import File, UploadFile

from common.constants import DATASETS_PATH
from common.utils import save_file

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import logging
import azure.functions as func
from api_app import app

try:
    from azure.functions import AsgiMiddleware
except ImportError:
    from functions._http_asgi import AsgiMiddleware

# Dynamically load all the models in to the API.
# Instead of that can manually add model routers in to the main app as app.include_router(router)
for subdir, dirs, files in os.walk("./models/"):
    for d in dirs:
        if not d.startswith("_") and not d.startswith("."):
            try:
                module = importlib.import_module('models.' + d)
                router = getattr(module, 'router')
                app.include_router(router)
                logging.info("Loaded model " + d)
            except Exception as e:
                logging.info("Failed to load model " + d + " " + str(e))


@app.get("/")
async def root():
    return {"message": "Hello SIDS ML Backend V6!"}


@app.get("/datasets")
async def list_datasets():
    return os.listdir(DATASETS_PATH)


@app.post("/upload_dataset/")
async def upload_files(files: List[UploadFile] = File(..., description="Multiple dataset file upload")):
    # in case you need the files saved, once they are uploaded
    for file in files:
        contents = await file.read()
        save_file(DATASETS_PATH + file.filename, contents)

    return {"Uploaded Filenames": [file.filename for file in files]}


def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    return AsgiMiddleware(app).handle(req, context)
