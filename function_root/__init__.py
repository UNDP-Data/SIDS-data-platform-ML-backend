import logging
import azure.functions as func

from api_app import app
from models.imputation import imputation
from models.timeseries import timeseries

try:
    from azure.functions import AsgiMiddleware
except ImportError:
    from functions._http_asgi import AsgiMiddleware


app.include_router(imputation.router)
app.include_router(timeseries.router)


@app.get("/")
async def root():
    return {"message": "Hello SIDS ML Backend!"}


def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    return AsgiMiddleware(app).handle(req, context)