import pandas as pd
import numpy as np
import logging
import os 

from fastapi import HTTPException

from common.constants import SIDS, DATASETS_PATH
from common.errors import Error
from common.logger import logger
from shared_dataloader.indicator_dataloader import data_importer
from shared_dataloader import data_loader

country_list = {}

# Import data
indicatorMeta = None
datasetMeta = None
indicatorData = None
time_estimate = None

def load_dataset():
    global indicatorMeta, datasetMeta, indicatorData
    indicatorMeta, datasetMeta, indicatorData = data_loader.load_data("twolvlImp", data_importer())
    indicatorData=indicatorData[indicatorData["Country Code"].isin(SIDS)]

load_dataset()

def country_options(target: str):
    global country_list
    if target not in country_list.keys():
        valid_countries = indicatorData[indicatorData['Indicator Code']==target]['Country Code'].values.tolist()
        country_list[target] = valid_countries
        #logging.info(indicatorData[indicatorData['Indicator Code']==target])

def check_country_validity(target,country):
    global country_list
    country_options(target)
    if ((country not in country_list[target]) | (len(country_list[target]) == 0)):
        raise HTTPException(status_code=422, detail=Error.INVALID_COUNTRY.format(country).value)

def get_country_list(target):
    global country_list
    country_options(target)
    value = indicatorData[indicatorData["Indicator Code"]==target]["Country Code"].values.tolist()
    return value

def imputer(indicator, country,method,direction='both',d=1,indicatorData=indicatorData):
    idx=pd.IndexSlice
    indicatorData = indicatorData.set_index(["Indicator Code","Country Code"]).sort_index(axis=1).dropna(axis=0,how='all')
    
    rename_names = dict()
    for i in indicatorData.columns:
        rename_names[i] = int(i)
    indicatorData.rename(columns=rename_names,inplace=True)    
    data = indicatorData.loc[[idx[indicator,country],]].copy()
    missing = data.columns[data.isnull().any()].tolist()
    if method in ["spline","polynomial"]:
        imp_data = data.copy().interpolate(method=method,order=d,axis=1,limit_direction=direction).dropna(axis=1, how='all')
    else:
        imp_data = data.copy().interpolate(method=method,axis=1,limit_direction=direction).dropna(axis=1, how='all').dropna(axis=1, how='all')
    missing_years = list(set(missing) & set(imp_data.columns))

    return missing_years,imp_data.to_dict(orient='split')

