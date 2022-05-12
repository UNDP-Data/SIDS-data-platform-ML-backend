import pandas as pd
import numpy as np
import logging

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,cross_val_predict

import os

from sklearn.svm import SVR, NuSVR
from sklearn.linear_model import SGDRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import Pool, CatBoostRegressor




from models.timeseriesImp.enums import Model, Interval
from common.constants import SIDS, DATASETS_PATH

from shared_dataloader.indicator_dataloader import data_importer
from shared_dataloader import data_loader
from common.errors import Error
from common.logger import logger
from fastapi import HTTPException

n= 10 #indicator cannot ignore more than this many sids countries
m = 5 #indicator must have measurements for atleast 1/mth of the total number of years in the indicatorData



def series_extractor(indicator, ind_data,method,direction='both',d=1):
    idx=pd.IndexSlice
    data = ind_data.loc[idx[indicator,SIDS],].copy()
    #missing = data.columns[data.isnull().any()].tolist()
    if method in ["spline","polynomial"]:
 
        imp_data = data.copy().interpolate(method=method,order=d,axis=1,limit_direction=direction).dropna(axis=1, how='all')
    else:
        imp_data = data.copy().interpolate(method=method,axis=1,limit_direction=direction).dropna(axis=1, how='all').dropna(axis=1, how='all')


    #missing_years = list(set(missing) & set(imp_data.columns))
    return imp_data


def missing_sids(indicator, ind_data):
    return len(list(set(SIDS)-set(ind_data.loc(axis=0)[pd.IndexSlice[indicator,]].index)))
    
def missing_years(indicator,ind_data):
    sumb = ind_data.loc(axis=0)[pd.IndexSlice[indicator,]]
    sumb=sumb.isna().sum(axis=0).sort_values()/sumb.shape[0]
    return sumb[sumb>0.5].index.shape[0]


def validity_check(ind_data,sids_count,years_count):
    ind_data = ind_data.set_index(["Indicator Code","Country Code"]).sort_index(axis=1).dropna(axis=0,how='all')#.interpolate('linear')

    indicators = []
    missing_sids_list = []
    missing_years_list = []
    actual_years_list = []
    for i in ind_data.index.levels[0]:
        indicators.append(i)

        try:
            missing_sids_list.append(missing_sids(i))
            c,y = missing_years(i)
            missing_years_list.append(c)
            actual_years_list.append(y)
        except:
            missing_sids_list.append(50)
            missing_years_list.append(50)
            actual_years_list.append([])
    validity = pd.DataFrame(data={"Indicator":indicators,"missing_sids":missing_sids_list,"missing_years":missing_years_list, "available_years":actual_years_list})
    return validity[(validity.missing_sids<sids_count) &(validity.missing_years<(len(indicatorData.columns)/years_count))]

def preprocessing(ind_data, predictors,target_year,target):

    ind_data = ind_data.set_index(["Indicator Code","Country Code"]).sort_index(axis=1).dropna(axis=0,how='all')#.interpolate('linear')

    rename_names = dict()
    for i in ind_data.columns:
        rename_names[i] = int(i)
        
    ind_data.rename(columns=rename_names,inplace=True)

    data = series_extractor(indicator=predictors,ind_data=ind_data,method='linear',direction="both")

    restructured_data = pd.DataFrame()

    sample_weight = pd.DataFrame()

    restructured_target = pd.DataFrame()

    window_counter =1
    year = target_year
    while (year-3) > min(2000,target_year-15):

        sub = data.loc(axis=1)[range(year-3,year)]

        sub = pd.DataFrame(data=sub.to_numpy(), index=sub.index,columns=["lag 3","lag 2","lag 1"]).unstack("Indicator Code").swaplevel(axis=1).sort_index(axis=1)
        sub["window"] = year
        sub.set_index('window',append=True,inplace=True)


        restructured_data = pd.concat([restructured_data,sub])
        weight= 1/window_counter
        sample_weight = pd.concat([sample_weight,pd.DataFrame(data=[weight]*sub.shape[0],index=sub.index,columns=["weight"])])
        window_counter = window_counter+1
        idx=pd.IndexSlice
        target_data = ind_data.loc[idx[target,SIDS],].copy()
        
        target_sub = target_data.loc(axis=1)[year]
        target_sub = pd.DataFrame(data=target_sub.to_numpy(), index=target_sub.index,columns=["target"]).unstack("Indicator Code").swaplevel(axis=1).sort_index(axis=1)
    #     for i in sub.index:
    #         if i not in target_sub.index:
    #             target_sub.loc[i,target_sub.columns[0]] = [np.nan]
        target_sub["window"] = year
        target_sub.set_index('window',append=True,inplace=True)
        restructured_target = pd.concat([restructured_target,target_sub])
        
        year = year-3
    
    restructured_data.dropna(axis=0,inplace=True)

    training_data = restructured_data.merge(restructured_target,how='left',left_index=True,right_index=True)
    training_data = training_data.merge(sample_weight,how='left',left_index=True,right_index=True)
    X_train= training_data[training_data[(target,"target")].notna()]
    X_test = training_data[training_data[(target,"target")].isna()]
    X_test.pop("weight")
    sample_weight = X_train.pop("weight")
    X_test.pop((target,"target"))
    y_train = X_train.pop((target,"target"))
    X_test = X_test.loc(axis=0)[pd.IndexSlice[:,target_year]]

    return X_train,X_test,y_train,sample_weight

def model_trainer(X_train, X_test, y_train, seed, n_estimators, model_type, interval,sample_weight):
    """
    Train the selected model, cross validate, score and generate a 90% prediction interval based on bootstrapped residuals.
    Args:
        X_train: training data
        X_test: prediction data
        y_train: training target array
        seed: random state setter
        n_estimators: number of trees for tree based models
        model: type of model to be trained
    Returns:
        prediction: dataframe with prediction values and confidence interval
        rmse: root mean squared error of the model
        gs: trained GridSearchCV model
        best_model: the best model
    """
    #X_train["sample_weight"]= X_train.reset_index()['Country Code'].apply(lambda x: 100 if x in SIDS else 1)

    model_list = None
    if model_type == Model.all:
        model_list = [e for e in Model if e != Model.all]
    else:
        model_list = [model_type]

    logging.info(model_list)
    model_instances = []
    params = []

    num_folds = min(5,X_train.shape[0])  # Hard coded
    scoring = 'neg_mean_squared_error'
    if Model.rfr in model_list:
        clf1 = RandomForestRegressor(random_state=seed)
        param1 = {}
        param1['regressor__n_estimators'] = [n_estimators]
        param1['regressor__max_depth'] = [5, 10, 20, 100, None]  # Hard coded
        param1['regressor'] = [clf1]
        model_instances.append(clf1)
        params.append(param1)
    if Model.etr in model_list:
        clf2 = ExtraTreesRegressor(random_state=seed)
        param2 = {}
        param2['regressor__n_estimators'] = [n_estimators]
        param2['regressor__max_depth'] = [5, 10, 20, 100, None]  # Hard coded
        param2['regressor'] = [clf2]
        model_instances.append(clf2)
        params.append(param2)

    if Model.gbr in model_list:
        clf3 = GradientBoostingRegressor(random_state=seed)
        param3 = {}
        if interval == Interval.quantile:
            param3['regressor__loss'] = ['quantile']
            param3['regressor__alpha'] = [0.5]  # hard coded
        param3['regressor__n_estimators'] = [n_estimators]
        param3['regressor__max_depth'] = [3, 5, 10, 20, None]  # Hard coded
        param3['regressor'] = [clf3]
        model_instances.append(clf3)
        params.append(param3)

    if Model.esvr in model_list:
        clf4 = SVR(kernel='linear')
        param4 = {}
        #param4['degree'] = [2,3,4]
        #param4['C'] = [1, 2, 3]  # Hard coded
        param4['regressor'] = [clf4]
        model_instances.append(clf4)
        params.append(param4)
    
    if Model.nusvr in model_list:
        clf5 = NuSVR(kernel='linear')
        param5 = {}
        #param5['degree'] = [2,3,4]
        #param5['C'] = [1, 2, 3]  # Hard coded
        param5['regressor'] = [clf5]
        model_instances.append(clf5)
        params.append(param5)
    
    if Model.sdg in model_list:
        clf6 = SGDRegressor()
        param6 = {}
        param6['penalty'] =['l2', 'l1', 'elasticnet']
        #param6['alpha'] = [0.0001,0.001,0.01,0.1]
        param6['regressor'] = [clf6]
        model_instances.append(clf6)
        params.append(param6)

    if Model.xgbr in model_list:
        clf7 = XGBRegressor(random_state=seed,importance_type='weight')
        param7 = {}
        param7['regressor__n_estimators'] = [n_estimators]
        param7['regressor__max_depth'] = [5, 10, 20, 100, None]  # Hard coded

        param7['regressor'] = [clf7]
        model_instances.append(clf7)
        params.append(param7)
    if Model.lgbmr in model_list:
        clf8 = LGBMRegressor(random_state=seed)
        param8 = {}
        param8['regressor__n_estimators'] = [n_estimators]
        param8['regressor__max_depth'] = [5, 10, 20, 100, None]  # Hard coded

        param8['regressor'] = [clf8]
        model_instances.append(clf8)
        params.append(param8)
    if Model.cat in model_list:
        clf9 = CatBoostRegressor(random_state=seed)
        param9 = {}
        #param9['regressor__n_estimators'] = [n_estimators]
        param9['regressor__max_depth'] = [5, 10]  # Hard coded

        param9['regressor'] = [clf9]
        model_instances.append(clf9)
        params.append(param9)


    pipeline = Pipeline([('regressor', model_instances[0])])

    n_jobs = 1
    if os.getenv("SEARCH_JOBS") is not None:
        n_jobs = int(os.getenv("SEARCH_JOBS"))
    

    logging.info("Perform grid search using %d jobs", n_jobs)

    kwargs = {pipeline.steps[-1][0] + '__sample_weight': sample_weight}
    gs = GridSearchCV(pipeline, params, cv=num_folds, n_jobs=n_jobs, scoring=scoring, refit=True).fit(X_train, y_train,**kwargs)
    rmse = np.sqrt(-gs.best_score_)

    best_model = gs.best_estimator_["regressor"]

    prediction = pd.DataFrame(gs.predict(X_test), columns=["prediction"], index=X_test.index)

    if interval == Interval.bootstrap.name:

        # Residual Bootsrapping  on validation data
        pred_train = cross_val_predict(best_model, X_train, y_train, cv=3)

        res = y_train - pred_train

        ### BOOTSTRAPPED INTERVALS ###

        alpha = 0.1  # (90% prediction interval) #Hard Coded

        bootstrap = np.asarray([np.random.choice(res, size=res.shape) for _ in range(100)])
        q_bootstrap = np.quantile(bootstrap, q=[alpha / 2, 1 - alpha / 2], axis=0)

        # prediction = pd.DataFrame(gs.predict(X_test), columns=["prediction"], index=X_test.index)
        prediction["upper"] = prediction["prediction"] + q_bootstrap[1].mean()
        prediction["lower"] = prediction["prediction"] + q_bootstrap[0].mean()

    else:
        if str(type(best_model)) == "<class 'sklearn.ensemble._gb.GradientBoostingRegressor'>":
            all_models = {}
            for alpha in [0.05, 0.95]:  # Hard Coded
                gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha,
                                                max_depth=gs.best_params_['regressor__max_depth'],
                                                n_estimators=gs.best_params_['regressor__n_estimators'])
                all_models["q %1.2f" % alpha] = gbr.fit(X_train, y_train)
                # For prediction

            prediction["lower"] = all_models["q 0.05"].predict(X_test)
            prediction["upper"] = all_models["q 0.95"].predict(X_test)
        else:
            pred_Q = pd.DataFrame()
            for pred in best_model.estimators_:
                temp = pd.Series(pred.predict(X_test))
                pred_Q = pd.concat([pred_Q, temp], axis=1)
            quantiles = [0.05, 0.95]  # Hard Coded

            for q in quantiles:
                s = pred_Q.quantile(q=q, axis=1)
                prediction[str(q)] = s.values
            prediction.rename(columns={"0.05": "lower", "0.95": "upper"}, inplace=True)  # Column names are hard coded

    # Predict for SIDS countries with missing values
    #prediction = prediction[prediction.index.isin(SIDS)]
    #prediction = prediction.reset_index().rename(columns={"index": "country"}).to_dict(orient='list')
    #################### Prediction dataframe and best_model instance are the final results of the ML################

    return prediction.droplevel(1), rmse, gs, best_model


# Import data
indicatorMeta = None
datasetMeta = None
indicatorData = None
time_estimate = None

def load_dataset():
    global indicatorMeta, datasetMeta, indicatorData
    indicatorMeta, datasetMeta, indicatorData = data_loader.load_data("twolvlImp", data_importer())


load_dataset()

global valid_predictors
valid_predictors = validity_check(indicatorData,n,m)

def predictor_validity(target: str):
    global valid_predictors
    predictor_list = valid_predictors.Indicator.values.tolist()
    if target in predictor_list:
        predictor_list.remove(target)
    return  predictor_list

global valid_targets
valid_targets = validity_check(indicatorData,n+5,m+5) #Temporary

def target_validity():
    global valid_targets
    return valid_targets.Indicator.values.tolist()

def year_validity(target: str):
    global valid_targets
    years = valid_targets[valid_targets.Indicator ==target].available_years.values[0]
    logging.info(years)
    return years

def train_predict(predictors, n_estimators, model_type,target,target_year,interval,ind_data=indicatorData):
    X_train,X_test,y_train,sample_weight = preprocessing(ind_data, predictors,target_year,target)
    seed = 7
    prediction, rmse, gs, best_model = model_trainer(X_train, X_test, y_train, seed, n_estimators, model_type, interval,sample_weight)
    try:
        feature_importance = best_model.coef_
    except:
        feature_importance = best_model.feature_importances_
    return prediction,rmse.item(), feature_importance.tolist(), X_train.columns.tolist()