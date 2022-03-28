#Data Manipulation
import logging
import os

import numpy as np
import pandas as pd
# Country name format
import pycountry
# Propcessing and training
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.feature_selection import RFE
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from common.constants import SIDS
from common.logger import logger

seed = 7

# Indicators measured for less than 90 percent of the years for each country (to be removed) for the time-series version of model
measure = 90

# Interpolation for indcators missing less than 30% using KNN imputer
percent = 30

def cou_ind_miss(Data):
    """
        Returns the amount of missingness across the years in each indicator-country pair
    """
    absolute_missing = Data.drop(columns=["Country Code","Indicator Code"]).isnull().sum(axis=1)
    total = Data.drop(columns=["Country Code","Indicator Code"]).count(axis=1)

    percent_missing = absolute_missing * 100 / Data.drop(columns=["Country Code","Indicator Code"]).shape[1]
    missing_value_df = pd.DataFrame({'row_name': Data["Country Code"]+"-"+Data["Indicator Code"],
                                 'Indicator Code':Data["Indicator Code"],
                                 'absolute_missing':absolute_missing,
                                 'total':total,
                                 'percent_missing': percent_missing})
    countyIndicator_missingness = missing_value_df.sort_values(["percent_missing","row_name"])

    return countyIndicator_missingness


def data_importer(percent=90, model_type="non-series", path="./datasets/"):
    """

        Import csv files and restrructure the data into a country by indcator format. Model_type will be expanded upon.
        precent: the most tolerable amount of missingness in a column for an indicator  accross the years
        model_type: type of model data imported for
        path: path on disk the raw data is stored

        wb_data: indicatorData restructed to a (country x year) by Indicator Code format
        indicatorMeta: indicator meta dataset (as is)
        indicatorData: indicator data dataset (as is)
        datasetMeta: dataset meta data (as is)
    """
    #

    logging.info('loading %s', path + "indicatorMeta.csv")

    try:
        indicatorMeta = pd.read_csv(path + "indicatorMeta.csv")
    except Exception as e:
        logging.exception("Read csv failed")


    logging.info('indicatorMeta.csv loaded %s', path + "indicatorMeta.csv")

    datasetMeta = pd.read_csv(path + "datasetMeta.csv")

    logging.info('datasetMeta.csv loaded')

    indicatorData = pd.read_csv(path + "indicatorData.csv")

    logging.info('indicatorData.csv loaded')

    #### Remove rows with missing country or indicator names
    indicatorData["Country/Indicator Code"] = indicatorData["Country Code"] + "-" + indicatorData["Indicator Code"]
    indicatorData = indicatorData[indicatorData["Country/Indicator Code"].notna()].drop(
        columns="Country/Indicator Code")

    if model_type == "series":
        # Indicators measured less than 5 times for each country are removed
        countyIndicator_missingness = cou_ind_miss(indicatorData)
        indicator_subset = set(countyIndicator_missingness[countyIndicator_missingness.percent_missing >= percent][
                                   "Indicator Code"]) - set(
            countyIndicator_missingness[countyIndicator_missingness.percent_missing < percent]["Indicator Code"])

        indicatorData = indicatorData[~indicatorData["Indicator Code"].isin(indicator_subset)]

    wb_data = indicatorData.set_index(['Country Code', 'Indicator Code'])
    wb_data = wb_data.stack()
    wb_data = wb_data.unstack(['Indicator Code'])
    wb_data = wb_data.sort_index()

    indicatorMeta = indicatorMeta[indicatorMeta["Indicator Code"].isin(indicatorData["Indicator Code"].values)]
    indicatorMeta = indicatorMeta[indicatorMeta.Indicator.notna()]

    datasetMeta = datasetMeta[datasetMeta["Dataset Code"].isin(indicatorMeta["Dataset"].values)]
    datasetMeta = datasetMeta[datasetMeta["Dataset Name"].notna()]

    return wb_data, indicatorMeta, datasetMeta, indicatorData


# Preprocess
def missingness(df):
    "Rank the columns of df by the amount of missing observations"
    absolute_missing = df.isnull().sum()
    percent_missing = absolute_missing * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                     'absolute_missing': absolute_missing,
                                     'percent_missing': percent_missing})
    return missing_value_df.sort_values(["percent_missing","column_name"])


def preprocessing(data, target, target_year, interpolator, SIDS, percent=30):
    """
    Preprocess data into a format suitable for the two step imputation model by filling the most complete
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
    if interpolator == 'KNNImputer':
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


    elif interpolator == 'SimpleImputer':
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
def feature_selector(X_train,y_train,manual_predictors):
    """
        Implement the Recursive feature selection for automatic selection of predictors for model

        returns: A boolean list of which features should be considered for prediction
    """
    # ESTIMATOR STILL UNDER INVESTIGATION: FOR NOW TAKE ON WITH HIGH DIMENSIONALITY TOLERANCE
    estimator= RandomForestRegressor()

    # STEP SIZE UNDER INVESTIGATION: FOR NOW TAKE ONE THAT REDUCES COMPUTATION TIME WITHOUT JUMPING
    selector = RFE(estimator,n_features_to_select=manual_predictors,step= manual_predictors)
    selector.fit(X_train,y_train)
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
        query_card: correlation plot of X_train (ignore for now)
    """

    if scheme == "Automatic via feature selection":

        # Take the most import predictor_number number of independent variables (via RFE) and plot correlation
        importance_boolean = feature_selector(X_train=X_train, y_train=y_train, manual_predictors=manual_predictors)
        prediction_features = (X_train.columns[importance_boolean].tolist())
        # query_card = correlation_plotter(target,prediction_features, training_data, ind_meta)
        X_train = X_train[prediction_features]
        X_test = X_test[prediction_features]

    elif scheme == "Automatic via PCA":
        pca = PCA(n_components=manual_predictors)
        pca.fit(X_train)
        columns = ["pca " + str(i) for i in list(range(manual_predictors))]
        X_train = pd.DataFrame(pca.transform(X_train), columns=columns, index=X_train.index)
        X_test = pd.DataFrame(pca.transform(X_test), columns=columns, index=X_test.index)
        #### Temporary ####
        # correlation= X_train.copy()
        # correlation[ind_meta[ind_meta["Indicator Code"]==target].Indicator.values[0]] = y_train.values
        # query_card = px.imshow(correlation.corr(),x=correlation.columns,y=correlation.columns, color_continuous_scale=px.colors.sequential.Blues, title= 'Correlation plot')


    else:
        prediction_features = manual_predictors
        # query_card = correlation_plotter(target,prediction_features, training_data, ind_meta)
        X_train = X_train[prediction_features]
        X_test = X_test[prediction_features]

    return X_train, X_test  # ,query_card

# Train model and predict
def model_trainer(X_train,X_test,y_train,seed,n_estimators, model,interval):
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

    if model == "all":
        model = ["rfr","etr","gbr","svr"]
    model_instances=[]
    params= []

    num_folds = 5 # Hard coded
    scoring = 'neg_mean_squared_error'
    if "rfr" in  model:
        clf1 = RandomForestRegressor(random_state = seed)
        param1 = {}
        param1['regressor__n_estimators'] = [n_estimators]
        param1['regressor__max_depth'] = [5, 10, 20,100, None] # Hard coded
        param1['regressor'] = [clf1]
        model_instances.append(clf1)
        params.append(param1)
    if "etr" in model:
        clf2 = ExtraTreesRegressor(random_state = seed)
        param2 = {}
        param2['regressor__n_estimators'] = [n_estimators]
        param2['regressor__max_depth'] = [5, 10, 20,100, None]# Hard coded
        param2['regressor'] = [clf2]
        model_instances.append(clf2)
        params.append(param2)

    if "gbr" in model:
        clf3 = GradientBoostingRegressor(random_state = seed)
        param3 = {}
        if interval == "quantile":
            param3['regressor__loss'] = ['quantile']
            param3['regressor__alpha'] = [0.5] # hard coded
        param3['regressor__n_estimators'] = [n_estimators]
        param3['regressor__max_depth'] = [3,5, 10, 20, None]# Hard coded
        param3['regressor'] = [clf3]
        model_instances.append(clf3)
        params.append(param3)

    pipeline = Pipeline([('regressor', model_instances[0])])

    n_jobs = 1
    if os.getenv("SEARCH_JOBS") is not None:
        n_jobs = int(os.getenv("SEARCH_JOBS"))

    logging.info("Perform grid search using %d jobs", n_jobs)
    gs = GridSearchCV(pipeline, params, cv=num_folds, n_jobs=n_jobs, scoring=scoring, refit=True).fit(X_train, y_train)
    rmse = np.sqrt(-gs.best_score_)

    best_model = gs.best_estimator_["regressor"]

    prediction = pd.DataFrame(gs.predict(X_test), columns=["prediction"], index=X_test.index)


    if interval == "bootstrap":

        #Residual Bootsrapping  on validation data
        pred_train = cross_val_predict(best_model,X_train, y_train, cv=3)

        res = y_train - pred_train

        ### BOOTSTRAPPED INTERVALS ###

        alpha = 0.1 #(90% prediction interval) #Hard Coded

        bootstrap = np.asarray([np.random.choice(res, size=res.shape) for _ in range(100)])
        q_bootstrap = np.quantile(bootstrap, q=[alpha/2, 1-alpha/2], axis=0)

        #prediction = pd.DataFrame(gs.predict(X_test), columns=["prediction"], index=X_test.index)
        prediction["upper"]= prediction["prediction"] + q_bootstrap[1].mean()
        prediction["lower"]= prediction["prediction"] + q_bootstrap[0].mean()

    else:
        if str(type(best_model))== "<class 'sklearn.ensemble._gb.GradientBoostingRegressor'>":
            all_models = {}
            for alpha in [0.05, 0.95]: # Hard Coded
                gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha,max_depth=gs.best_params_['regressor__max_depth'],n_estimators=gs.best_params_['regressor__n_estimators'])
                all_models["q %1.2f" % alpha] = gbr.fit(X_train, y_train)
                #For prediction

            prediction["lower"]= all_models["q 0.05"].predict(X_test)
            prediction["upper"]= all_models["q 0.95"].predict(X_test)
        else:
            pred_Q = pd.DataFrame()
            for pred in best_model.estimators_:
                temp = pd.Series(pred.predict(X_test))
                pred_Q = pd.concat([pred_Q,temp],axis=1)
            quantiles = [0.05, 0.95] # Hard Coded

            for q in quantiles:
                s = pred_Q.quantile(q=q, axis=1)
                prediction[str(q)] = s.values
            prediction.rename(columns={"0.05":"lower","0.95":"upper"}, inplace=True) # Column names are hard coded

    # Predict for SIDS countries with missing values
    prediction = prediction[prediction.index.isin(SIDS)]
    prediction.index = [pycountry.countries.get(alpha_3=i).name for i in prediction.index]
    prediction = prediction.reset_index().rename(columns={"index":"country"})
    #################### Prediction dataframe and best_model instance are the final results of the ML################

    return prediction,rmse,gs, best_model

# Import data
wb_data = None
indicatorMeta = None
datasetMeta = None
indicatorData = None


def query_and_train(manual_predictors, target_year, target,interpolator,scheme,estimators,model,interval, ind_meta):
    global wb_data, indicatorMeta, datasetMeta, indicatorData
    if wb_data is None:
        wb_data, indicatorMeta, datasetMeta, indicatorData = data_importer(model_type="knn")

    if ind_meta is None:
        ind_meta = indicatorMeta

    logging.info('Data set loaded')

    # Train test (for prediction not validation) split
    X_train, X_test, y_train = preprocessing(data=indicatorData, target=target, target_year=target_year,
                                             interpolator=interpolator, SIDS=SIDS, percent=percent)

    logging.info('Data preprocessed')
    # Dimension reduction based on scheme
    X_train, X_test = feature_selection(X_train, X_test, y_train, target, manual_predictors, scheme)

    logging.info('Feature selection completed')
    # training and prediction for X_test
    prediction, rmse, gs, best_model = model_trainer(X_train, X_test, y_train, seed, estimators, model, interval)

    return (rmse/y_train.mean()).item(), rmse.item(), best_model.feature_importances_.tolist(), best_model.feature_names_in_.tolist(), prediction