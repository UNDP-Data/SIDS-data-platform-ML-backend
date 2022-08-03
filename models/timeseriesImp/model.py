import pandas as pd
import numpy as np
import logging

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,cross_val_predict

import os

from sklearn.svm import SVR, NuSVR
from sklearn.linear_model import SGDRegressor





from models.timeseriesImp.enums import Model, Interval
from common.constants import SIDS, DATASETS_PATH

from shared_dataloader.indicator_dataloader import data_importer
from shared_dataloader import data_loader
from common.errors import Error
from common.logger import logger
from fastapi import HTTPException

n= 15 #indicator cannot ignore more than this many sids countries
m = 2 #indicator must have measurements for atleast 1/mth of the total number of years in the indicatorData



def series_extractor(indicator, ind_data,method,direction='both',d=1):
    """
        Interpolate ind_data using pandas interpolate method for filling missing timerseries data
        Args:
            indicator: indicator Code 
            ind_data: indicatorData dataset
            method: interpolation method. Options explained on https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html
            direction: direction of filling missing values. explained onhttps://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html
            d: order of polynomial for 'spline' and 'ploynomial' methods.
        returns:
            imp_data: interpolated indicatorData dataset for "indicator" and SIDS 
    
    """
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
    """Calculates the number of SIDS that are never measured for this indicator"""

    return len(list(set(SIDS)-set(ind_data.loc(axis=0)[pd.IndexSlice[indicator,]].index)))
    
def missing_years(indicator,ind_data):
    """Calcuates the number of missing years for this indicator
    returns:
        missing years count 
        missing years as list
        actual years for which the indicator is observed
    """
    sumb = ind_data.loc(axis=0)[pd.IndexSlice[indicator,]]
    sumb=sumb.isna().sum(axis=0).sort_values()/sumb.shape[0]
    return sumb[sumb>0.5].index.shape[0]


def validity_check(ind_data,sids_count,years_count):

    """ Returns indicators which satisfy certain amount of non-missing data
    Args:
        ind_data: indicatorData dataset
        sids_count: threshold determining number of SIDS that are never measured for an indicator
        years_count: indicator must have measurements for atleast 1/year_count of the total number of years in the indicatorData
    Returns:
        dataframe with valid indicators according to the threshold in the input arguments
    """
    #ind_data = ind_data.set_index(["Indicator Code","Country Code"]).sort_index(axis=1).dropna(axis=0,how='all')#.interpolate('linear')

    indicators = []
    missing_sids_list = []
    missing_years_list = []
    missing_years_count_list = []
    actual_years_list = []
    try:
        indList = ind_data.index.levels[0]
    except:
        ind_data = ind_data.set_index(["Indicator Code","Country Code"]).sort_index(axis=1).dropna(axis=0,how='all')
        indList = ind_data.index.levels[0]
    for i in indList:
        indicators.append(i)

        try:
            missing_sids_list.append(missing_sids(i))
            c,my,y = missing_years(i)
            missing_years_count_list.append(c)
            missing_years_list.append(my)
            actual_years_list.append(y)
        except:
            missing_sids_list.append(50)
            missing_years_count_list.append(50)
            actual_years_list.append([])
            missing_years_list.append([])
    validity = pd.DataFrame(data={"Indicator":indicators,"missing_sids":missing_sids_list,"missing_years_count":missing_years_count_list,"missing_years":missing_years_list, "available_years":actual_years_list})
    return validity[(validity.missing_sids<sids_count) &(validity.missing_years_count<(len(ind_data.columns)/years_count))]

def preprocessing(ind_data, predictors,target_year,target):

    """
    Transform the ind_data dataframe into a (country, year) by (indicator Code) by  creating the window and lag features.
    the sliced data frame is reshaped into an multi-index data frame where each row represents a country and target history (also called window) pair. Each predictor column, on the other hand, represents the historical information (also called lag) of the indicator.
    In addition, sample weight is generated such that target history (windows) further away from the target year are given small weights to force the model to focus on the relationship between predictors and indicators close in time to the target year.
    For e.g.: For window 2, lag 3 and target year =2010, the dataframe generated looks like (where the values are represented by the corresponding years)
                      Indicator/predictor    target
                      lag3 lag2 lag1
    country1 window1  2007 2008 2009         2010
             window2  2006 2007 2008         2009
    country2 window1  2007 2008 2009         2010
             window2  2006 2007 2008         2009
    Args:
        ind_data: indicatorData
        predictors: list of predictors indicator codes
        target_year: the target year under consideration for imputation
        target: the indicator to be imputed
    Returns:
        X_train: subset of generated reshaped data where target is measured
        X_test: subset of  generated reshaped data where target for target year is missing
        y_train: X_train's corresponding target pandas series
        sample_weight: pandas series with weights for the X_train observations/rows
    
    """

    data = series_extractor(indicator=predictors,ind_data=ind_data,method='linear',direction="both")

    restructured_data = pd.DataFrame()

    sample_weight = pd.DataFrame()

    restructured_target = pd.DataFrame()

    window_counter =1
    year = target_year
    while (year-3) > min(2000,target_year-15): # hard coded

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
        
        year = year-1
    
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
valid_targets = validity_check(indicatorData,50,1.25) #Temporary

def target_validity():
    global valid_targets
    return valid_targets.Indicator.values.tolist()

def year_validity(target: str):
    global valid_targets
    years = valid_targets[valid_targets.Indicator ==target].available_years.values[0]
    logging.info(years)
    return years

def train_predict(predictors, n_estimators, model_type,target,target_year,interval,ind_data=indicatorData):

    data = ind_data.set_index(["Indicator Code","Country Code"]).sort_index(axis=1).dropna(axis=0,how='all')#.interpolate('linear')

    rename_names = dict()
    for i in data.columns:
        rename_names[i] = int(i)
        
    data.rename(columns=rename_names,inplace=True)
    X_train,X_test,y_train,sample_weight = preprocessing(data, predictors,target_year,target)
    seed = 7
    prediction, rmse, gs, best_model = model_trainer(X_train, X_test, y_train, seed, n_estimators, model_type, interval,sample_weight)
    try:
        feature_importance = best_model.coef_
    except:
        feature_importance = best_model.feature_importances_

    indicator_importance = pd.DataFrame(data={"names":X_train.columns.tolist(), "values":feature_importance})

    indicator_importance['predictor'] = indicator_importance["names"].apply(lambda x: x[0])

    #indicator_importance.sort_values(["values"],inplace=True, ascending=False)
    importanceSummed = indicator_importance.groupby(['predictor']).sum()
    importanceSorted = importanceSummed.reset_index().sort_values('values',ascending = False).head(10)
    #indicator_importance = importanceSorted.sort_values(["year","target","values"], ascending=False)
    
    SI_index = rmse / y_train.mean()
    if ((SI_index >1) | (SI_index<0)):
        SI_index= rmse / (y_train.max()-y_train.min())
    return prediction,SI_index, importanceSorted["values"].values.tolist(),importanceSorted["predictor"].values.tolist() #feature_importance.tolist(), X_train.columns.tolist()