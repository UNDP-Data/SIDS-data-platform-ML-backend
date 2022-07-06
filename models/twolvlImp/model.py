# Data Manipulation
import logging
import os
import math
import numpy as np
import pandas as pd

# Propcessing and training
from fastapi import HTTPException
from pca import pca
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.feature_selection import RFE
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from common.constants import SIDS, DATASETS_PATH
from common.errors import Error
from common.logger import logger
from shared_dataloader import data_loader
from models.twolvlImp import Schema, Interpolator, Interval, Model
from shared_dataloader.indicator_dataloader import data_importer

seed = 7

#Maximum percentage of missingness to consider in target (target threshold) 
percent = 80 

#Maximum number of missingess to consider in predictors (predictor threshold)
measure = 30

supported_years = [str(x) for x in list(range(2000, 2020))]
Datasets = {}
Targets_top_ranked = {}
Predictor_list = {}

# Preprocess
def missingness(df):
    "Rank the columns of df by the amount of missing observations"
    absolute_missing = df.isnull().sum()
    percent_missing = absolute_missing * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                     'absolute_missing': absolute_missing,
                                     'percent_missing': percent_missing})
    return missing_value_df.sort_values(["percent_missing", "column_name"])


def preprocessing(data, target, target_year, interpolator, SIDS, percent=30):
    """
    Preprocess data into a format suitable for the two step twolvlImp model by filling the most complete
    Args:
        data: indicatorData dataset
        target: indicator whose values will be imputed
        target_year: the year under consideration
        interpolator: type of imputer to use for interpolation
        precent: the most tolerable amount of missingness in a column

    Returns:
        X_train: training data
        X_test:  testing data (observation with missing values for target variable)
        y_train: training target series
    """

    # Subset data for target and target_year
    # if str(target_year) not in data.index:
    #     raise HTTPException(status_code=404, detail={"msg": "Given rget year not in the dataset"})
    data_sub = data[["Country Code", "Indicator Code", str(target_year)]]
    data_sub = data_sub.set_index(["Country Code", "Indicator Code"])[str(target_year)].unstack(level=1)

    X_train = data_sub[data_sub[target].notna()]
    X_test = data_sub[(data_sub[target].isna()) & (data_sub.index.isin(SIDS))]

    y_train = X_train.pop(target)
    y_test = X_test.pop(target)

    # Find how much missing values are there for each indicator
    rank = missingness(X_train)

    # interpolation for indcators missing less than percent% using KNN imputer
    most_complete = rank[rank.percent_missing < percent]["column_name"].values

    X_train = X_train[most_complete]
    X_test = X_test[most_complete]

    # How muc does fiting only on X_train affect fits (perhaps another layer of performance via CV)
    if interpolator == Interpolator.KNNImputer:
        scaler = MinMaxScaler()
        imputer = KNNImputer(n_neighbors=5)  # Hard Coded
        scaler.fit(X_train)
        imputer.fit(scaler.transform(X_train))

        X_train = pd.DataFrame(data=scaler.inverse_transform(imputer.transform(scaler.transform(X_train)))
                               , columns=X_train.columns,
                               index=X_train.index)

        X_test = pd.DataFrame(data=scaler.inverse_transform(imputer.transform(scaler.transform(X_test)))
                              , columns=X_test.columns,
                              index=X_test.index)


    elif interpolator == Interpolator.SimpleImputer:
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')  # Hard coded
        imp_mean.fit(X_train)
        X_train = pd.DataFrame(data=imp_mean.transform(X_train)
                               , columns=X_train.columns,
                               index=X_train.index)
        X_test = pd.DataFrame(data=imp_mean.transform(X_test)
                              , columns=X_test.columns,
                              index=X_test.index)

    else:
        imp = IterativeImputer(missing_values=np.nan, random_state=0,
                               estimator=ExtraTreesRegressor(n_estimators=10, random_state=0), n_nearest_features=100,
                               add_indicator=False, sample_posterior=False)  # Hard Coded values
        imp.fit(X_train)
        X_train = pd.DataFrame(data=imp.transform(X_train)
                               , columns=X_train.columns,
                               index=X_train.index)
        X_test = pd.DataFrame(data=imp.transform(X_test)
                              , columns=X_test.columns,
                              index=X_test.index)

    return X_train, X_test, y_train


# Select features/ reduce dimensionality
def feature_selector(X_train, y_train, manual_predictors):
    """
        Implement the Recursive feature selection for automatic selection of predictors for model

        returns: A boolean list of which features should be considered for prediction
    """
    # ESTIMATOR STILL UNDER INVESTIGATION: FOR NOW TAKE ON WITH HIGH DIMENSIONALITY TOLERANCE
    estimator = RandomForestRegressor()

    # STEP SIZE UNDER INVESTIGATION: FOR NOW TAKE ONE THAT REDUCES COMPUTATION TIME WITHOUT JUMPING
    selector = RFE(estimator, n_features_to_select=manual_predictors, step=manual_predictors)
    selector.fit(X_train, y_train)
    return selector.support_


def feature_selection(X_train, X_test, y_train, target, manual_predictors, scheme):
    """
        Returns the training and testing/prediction data with a reduced feature space
    Args:
        X_train: training data
        X_test: prediction data
        y_train: training target array
        target: name of target array
        manual_predictors: number of predictors (for Automatic) or list of predictors (for Manual)
        scheme: feature selection method selected by user
    Returns:
        X_train: reduced training data
        X_test: reduced testing data
        correlation: correlation matrix of X_train (ignore for now)
    """

    if scheme == Schema.AFS:
        # Take the most import predictor_number number of independent variables (via RFE) and plot correlation
        importance_boolean = feature_selector(X_train=X_train, y_train=y_train, manual_predictors=manual_predictors)
        prediction_features = (X_train.columns[importance_boolean].tolist())


    elif scheme == Schema.PCA:
        PCA = pca()
        out = PCA.fit_transform(X_train)
        prediction_features = list(out['topfeat'].iloc[list(range(manual_predictors)),1].values)

    else:
        prediction_features = manual_predictors


    X_train = X_train[prediction_features]
    X_test = X_test[prediction_features]

    correlation = X_train[prediction_features].corr()
    correlation.index = prediction_features
    correlation.columns = prediction_features

    return X_train, X_test,correlation.to_dict(orient ='split')


# Train model and predict
def model_trainer(X_train, X_test, y_train, seed, n_estimators, model_type, interval):
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

    sample_weight = []#X_train.pop("sample_weight")
    sids_weights = (X_train.index.isin(SIDS)).sum()
    total = X_train.shape[0]

    # Inverse class weighting for SIDS and non-SIDS
    for i in X_train.index:
        if i in SIDS:
            sample_weight.append(1/sids_weights)
        else:
            sample_weight.append(1/(total-sids_weights))
    model_list = None
    if model_type == Model.all:
        model_list = [e for e in Model if e != Model.all]
    else:
        model_list = [model_type]

    logging.info(model_list)
    model_instances = []
    params = []

    num_folds = 5  # Hard coded
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
    prediction = prediction[prediction.index.isin(SIDS)]
    prediction = prediction.reset_index().rename(columns={"index": "country"})#.to_dict(orient='list')
    #################### Prediction dataframe and best_model instance are the final results of the ML################

    return prediction, rmse, gs, best_model

def sids_top_ranked(target_year,data,SIDS, percent,indicator_type="target"):
    """Return a list with indicators with less than "percent" amount of missingness over SIDS COUNTRIES only """
    sub_data = data[["Country Code","Indicator Code",str(target_year)]]
    sub_data = sub_data[sub_data["Country Code"].isin(SIDS)]
    sub_data = sub_data.set_index(["Country Code","Indicator Code"])[str(target_year)].unstack(level=1)
    rank = missingness(sub_data)
    if indicator_type == "target":
        top_ranked = rank[(rank.percent_missing < percent)&(rank.percent_missing > 0)]["column_name"].values
    else:
        top_ranked = rank[(rank.percent_missing < percent)]["column_name"].values
    return top_ranked

def total_top_ranked(target_year,data,SIDS, percent,indicator_type="target"):
    """Return a list with indicators with less than "percent" amount of missingness over ALL COUNTRIES (SIDS input not used)"""
    sub_data = data[["Country Code","Indicator Code",str(target_year)]]
    sub_data = sub_data.set_index(["Country Code","Indicator Code"])[str(target_year)].unstack(level=1)
    rank = missingness(sub_data)
    if indicator_type == "target":
        top_ranked = rank[(rank.percent_missing < percent)&(rank.percent_missing > 0)]["column_name"].values
    else:
        top_ranked = rank[(rank.percent_missing < percent)]["column_name"].values
    return top_ranked

def replacement(target,year, ind_data, ind_meta, sids, pred):
    idx= pd.IndexSlice
    subset_data = ind_data[ind_data["Indicator Code"]==target][["Country Code","Indicator Code",str(year)]].set_index(["Country Code","Indicator Code"]).stack(dropna=False).unstack("Indicator Code")
    subset_data = subset_data.loc[idx[sids,:],:]
    sub_pred= pred
    try:
        assert subset_data.isna().sum().sum() > 0
    except:
        print("no missing data found")
    results = subset_data.copy()
    lower = subset_data.copy()                                                                                                                                  
    upper = subset_data.copy()                                                                                                                                 
    for i in subset_data.index:
                for j in subset_data.columns:
                    value = subset_data.loc[i,j]
                    if np.isnan(value):

                        results.loc[i,j] = sub_pred[(sub_pred["Country Code"]==i[0])].prediction.values[0]#sub_data.loc[i,j]
                        lower.loc[i,j] = sub_pred[(sub_pred["Country Code"]==i[0])].lower.values[0]
                        upper.loc[i,j] = sub_pred[(sub_pred["Country Code"]==i[0])].upper.values[0]
                                                                                                                                  
                    else:
                        lower.loc[i,j] = np.nan
                        upper.loc[i,j] = np.nan
    for i in np.unique(pred["Country Code"]).tolist():
        try:
            assert i in results.index.levels[0], f"cannot find "+ i + " for year " + str(year)
        except:
            missed = pred[(pred["Country Code"]==i)]
            p = pd.DataFrame(data = [missed.prediction.values], columns=[target], index=[(i,year)])
            l = pd.DataFrame(data = [missed.lower.values], columns=[target], index=[(i,year)])
            u = pd.DataFrame(data = [missed.upper.values], columns=[target], index=[(i,year)])
            results=pd.concat([results,p])
            lower=pd.concat([lower,l])
            upper=pd.concat([upper,u])
    return results,lower,upper                                                                  

def processMLData(result,lower,upper,featureImportancesDf,year,target,cor,rmse,rmse_deviation):
    predictionsDf = result.reset_index().rename(columns={"level_1":"year"})
    lowerIntervalsDf = lower.reset_index().rename(columns={"level_1":"year"})
    upperIntervalsDf = upper.reset_index().rename(columns={"level_1":"year"})
    yearValues={}
    upperIntervals={}
    lowerIntervals={}
    indicatorJson={"data":{},"upperIntervals":{},"lowerIntervals":{},"categoryImportances":{},"featureImportances":{},
                  "correlation":{},"rmse":{},"rmse_deviation":{}}

    countries=predictionsDf["Country Code"].unique().tolist()
    for country in countries:
        value=predictionsDf[predictionsDf["Country Code"]==country][target].iloc[0]
        lower=lowerIntervalsDf[lowerIntervalsDf["Country Code"]==country][target].iloc[0]
        upper=upperIntervalsDf[upperIntervalsDf["Country Code"]==country][target].iloc[0]

        if not pd.isna(value):
            yearValues[country]=value

        if not pd.isna(lower):
            lowerIntervals[country]=lower

        if not pd.isna(upper):
            upperIntervals[country]=upper                       

    indicatorFeaturesDf=featureImportancesDf[featureImportancesDf["predicted indicator"]==target]
    features=indicatorFeaturesDf["feature indicator"].unique().tolist()
    featureImportances={}
    for feature in features:
        featureImportance=indicatorFeaturesDf[indicatorFeaturesDf["feature indicator"]==feature]["feature importance"].iloc[0]
        featureImportances[feature]=featureImportance

    featuresMeta=indicatorMeta[indicatorMeta["Indicator Code"].isin(features)]
    categories=featuresMeta["Category"].unique().tolist()

    categoryImportances={}

    for category in categories:
        categoryTotal=0
        for feature in featuresMeta[featuresMeta["Category"]==category]["Indicator Code"].unique().tolist():
            importance=featureImportances[feature]
            categoryTotal+=importance
        categoryImportances[category]=categoryTotal

    indicatorJson["data"][year]=yearValues
    indicatorJson["upperIntervals"][year]=upperIntervals
    indicatorJson["lowerIntervals"][year]=lowerIntervals
    indicatorJson["featureImportances"][year]=featureImportances
    indicatorJson["categoryImportances"][year]=categoryImportances
    indicatorJson["correlation"][year] = cor
    indicatorJson["rmse"][year] = rmse
    indicatorJson["rmse_deviation"][year] = rmse_deviation
    
    return indicatorJson



# Import data
indicatorMeta = None
datasetMeta = None
indicatorData = None
time_estimate = None

def load_dataset():
    global indicatorMeta, datasetMeta, indicatorData
    indicatorMeta, datasetMeta, indicatorData = data_loader.load_data("twolvlImp", data_importer())


load_dataset()

def load_time_estimate():
    global time_estimate
    try:
        time_estimate = pd.read_json(DATASETS_PATH+"time_consumption")
        time_estimate = pd.json_normalize(time_estimate['value'])
        logging.info('time estimate loaded')
    except Exception as e:
        logging.exception("Read time_estimate failed: " + str(e))

load_time_estimate()


def dimension_options(target, target_year):
    top_ranked = total_top_ranked(target_year,indicatorData,SIDS, percent)
    codes = indicatorMeta[indicatorMeta["Indicator"] == target][["Indicator Code", "Dimension"]]
    codes = codes[codes["Indicator Code"].isin(top_ranked)]
    return [{'label': codes.loc[i, "Dimension"], 'value': codes.loc[i, "Indicator Code"]} for i in codes.index]


def check_dataset_validity(target_year, dataset):
    options = dataset_options(target_year)
    if options.get(dataset) is None:
        raise HTTPException(status_code=422, detail=Error.INVALID_DATASET.format(list(options.keys())).value)


def dataset_options(target_year):
    logging.info("Dataset options:")
    # top_ranked = indicator_list(target_year,data,SIDS, percent=percent)
    global Datasets
    if target_year not in Datasets:
        logging.info("Dataset does not found in the cache for year %s", target_year)
        top_ranked = total_top_ranked(target_year,indicatorData,SIDS, percent)
        Datasets[target_year] = np.unique(indicatorMeta[indicatorMeta["Indicator Code"].isin(top_ranked)].Dataset.values)
    data_list = {}
    for i in Datasets[target_year]:
        names = datasetMeta[datasetMeta["Dataset Code"] == i]["Dataset Name"]
        if len(names.values) > 0:
            data_list[i] = names.values[0]
    return data_list


def get_target_years():
    return supported_years


def check_year_validity(year):
    if year not in supported_years:
        raise HTTPException(status_code=422, detail=Error.INVALID_TARGET_YEAR.format(supported_years[0], supported_years[-1]).value)


def load_predictos(target_year: str):
    global Predictor_list
    if target_year not in Predictor_list:
        top_ranked = total_top_ranked(target_year,indicatorData,SIDS, measure,indicator_type="predictor")
        Predictor_list[target_year] = top_ranked


def check_predictors_validity(target_year, predictors):
    global Predictor_list
    load_predictos(target_year)
    invalid_predictors = []

    for p in predictors:
        res = np.where(Predictor_list[target_year] == p)
        if res[0].size <= 0:
            invalid_predictors.append(p)

    logging.info(invalid_predictors)
    if len(invalid_predictors) > 0:
        raise HTTPException(status_code=422, detail=Error.INVALID_PREDICTOR.format(invalid_predictors).value)


def get_predictor_list(target, target_year, scheme):
    if scheme == Schema.MANUAL:

        global Predictor_list
        load_predictos(target_year)

        # top_ranked = indicator_list(target_year,data,SIDS, percent=percent)
        top_ranked = Predictor_list[target_year]
        if target in target_year:
            top_ranked = np.delete(top_ranked, np.where(top_ranked == target))
        names = indicatorMeta[indicatorMeta["Indicator Code"].isin(top_ranked)][
            ["Indicator", "Dimension", "Indicator Code"]].set_index("Indicator Code")
        return {i: (str(names.loc[i, "Indicator"]) + "-" + str(names.loc[i, "Dimension"])) for i in names.index}
    else:
        return {i: i for i in list(range(10, 51))}


def check_target_validity(target_year: str, dataset: str, target: str):
    key = target_year + "_" + dataset
    global Targets_top_ranked
    load_indicator(target_year, dataset)
    if key not in Targets_top_ranked or target not in Targets_top_ranked[key]:
        raise HTTPException(status_code=422, detail=Error.INVALID_TARGET.value)


def load_indicator(target_year: str, dataset: str):
    key = target_year + "_" + dataset
    global Targets_top_ranked
    if key not in Targets_top_ranked:
        logging.info("Indicator list does not found in the cache for key %s", key)
        Targets_top_ranked[key] = total_top_ranked(target_year,indicatorData,SIDS, percent)




def get_indicator_list(target_year: str, dataset: str):
    key = target_year + "_" + dataset
    global Targets_top_ranked
    load_indicator(target_year, dataset)
    return (indicatorMeta[indicatorMeta["Indicator Code"].isin(Targets_top_ranked[key])].groupby("Indicator Code").nth(0))["Indicator"].to_dict()


def target_sample_size_requirement():
    return math.ceil(indicatorData["Country Code"].unique().shape[0]* (1-percent/100))

def predictor_sample_size_requirement():
    return math.ceil(indicatorData["Country Code"].unique().shape[0]* (1-measure/100))

def get_time_estimate(scheme_name):
    if scheme_name.name not in time_estimate.scheme_e.values:
        return 2*60
    else:
        return time_estimate[time_estimate.scheme_e==scheme_name.name].avg_time_parsed.values[0]




def query_and_train(manual_predictors, target_year, target, interpolator, scheme, estimators, model, interval,
                    ind_meta):
    logging.info('Data set loaded')
    # Train test (for prediction not validation) split
    X_train, X_test, y_train = preprocessing(data=indicatorData, target=target, target_year=target_year,
                                             interpolator=interpolator, SIDS=SIDS, percent=measure)

    logging.info('Data preprocessed')
    # Dimension reduction based on scheme
    X_train, X_test,correlation = feature_selection(X_train, X_test, y_train, target, manual_predictors, scheme)

    logging.info('Feature selection completed')
    # training and prediction for X_test
    prediction, rmse, gs, best_model = model_trainer(X_train, X_test, y_train, seed, estimators, model, interval)
    
    # data for pie chart for feature importance 
    #features = indicatorMeta[indicatorMeta["Indicator Code"].isin(X_train.columns)]
    #feature_importance_pie =pd.DataFrame(data={"category":features.Category.values,"value":best_model.feature_importances_}).groupby("category").sum().reset_index().to_dict(orient="list")
    SI_index = rmse / y_train.mean()
    if ((SI_index >1) | (SI_index<0)):
        SI_index= rmse / (y_train.max()-y_train.min())
    
    logging.info("initiate change of features")
    importance = pd.DataFrame({"predicted indicator":[target]*len(best_model.feature_names_in_.tolist()),"feature indicator": best_model.feature_names_in_.tolist(),"feature importance":best_model.feature_importances_.tolist()})
    results,lower,upper = replacement(target=target,year=target_year, ind_data=indicatorData, ind_meta=indicatorMeta, sids=SIDS, pred=prediction)
    indicatorJson = processMLData(result=results,lower=lower,upper=upper,featureImportancesDf=importance,year=target_year,target=target,cor=correlation,rmse=rmse,rmse_deviation=SI_index)
    return SI_index.item(), rmse.item(),indicatorJson
    
    #return SI_index.item(), rmse.item(), best_model.feature_importances_.tolist(), best_model.feature_names_in_.tolist(), prediction.to_dict(orient='list'),correlation,feature_importance_pie
