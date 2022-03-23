import time
from textwrap import wrap

# For plotting
import plotly.express as px
#import plotly.graph_objs as go


# For helper functions
from data.constants import SIDS
from models.imputation.models import model_trainer
from utils import *

# For data manipulation
import pandas as pd
from sklearn.decomposition import PCA


import pycountry

# connect to app

#Maximum percentage to consider
percent=80


# Import data
wb_data,indicatorMeta, datasetMeta,indicatorData = data_importer(model_type="knn")

def query_and_train(manual_predictors, target_year, target,interpolator,scheme,n_estimators,model,interval, ind_meta=indicatorMeta):

    print(target)
    t0 = time.time()

    # Setup interpolated training data using helper function from utils
    training_data, filled_data = preprocessing(data=indicatorData,target=target,target_year=target_year,interpolator=interpolator,percent=80)
    print("data preprocessed")

    # Train test split based on missingness in target variable
    X_train = filled_data[training_data[target].notna()].copy()
    X_test = filled_data[(training_data[target].isna())&(training_data.index.isin(SIDS))].copy()
    #X_test = filled_data[(training_data[target].isna())]
    y_train = training_data[target][training_data[target].notna()]
    y_test = training_data[target][training_data[target].isna()]
    print("data split")

    # Make sure target variable is not in training data predicion features
    if target in X_train.columns:
        X_train.pop(target)
        X_test.pop(target)

    print("target poped")

    # Caluclate the average value for target variable
    value_for_si = y_train.mean()

    if scheme == "Automatic via feature selection":

        # Take the most import predictor_number number of independent variables (via RFE) and plot correlation
        importance_boolean = feature_selector(X_train=X_train,y_train=y_train,manual_predictors=manual_predictors)
        prediction_features = (X_train.columns[importance_boolean].tolist())
        query_card = correlation_plotter(target,prediction_features, training_data, ind_meta)
        if target in prediction_features:
            prediction_features.remove(target)
        X_train = X_train[prediction_features]
        X_test = X_test[prediction_features]

    elif scheme=="Automatic via PCA":
        pca = PCA(n_components=manual_predictors)
        pca.fit(X_train)
        columns = ["pca "+ str(i) for i in list(range(manual_predictors))]
        X_train = pd.DataFrame(pca.transform(X_train), columns=columns, index=X_train.index)
        X_test = pd.DataFrame(pca.transform(X_test), columns=columns, index=X_test.index)
        #### Temporary ####
        correlation= X_train.copy()
        correlation[ind_meta[ind_meta["Indicator Code"]==target].Indicator.values[0]] = y_train.values
        query_card = px.imshow(correlation.corr(),x=correlation.columns,y=correlation.columns, color_continuous_scale=px.colors.sequential.Blues, title= 'Correlation plot')


    else:
        prediction_features= manual_predictors
        query_card = correlation_plotter(target,prediction_features, training_data, ind_meta)
        X_train = X_train[prediction_features]
        X_test = X_test[prediction_features]

    seed=7


        #alert,SI_alert, avp_fig, coef_fig, bar_fig,bar_fig_1,bar_fig_2 = model_trainer(X_train,X_test,y_train,seed,n_estimators,t0,target_year,target,model,interval)


        #return alert,SI_alert, avp_fig, coef_fig, query_card, bar_fig,bar_fig_1,bar_fig_2
    prediction,rmse,gs, best_model = model_trainer(X_train,X_test,y_train,seed,n_estimators,t0,target_year,target,model,interval)
	# Plot feature importance
	
    #if str(type(best_model)) == "<class 'sklearn.svm._classes.SVR'>":
    #    print("svr")
    #    coef_fig = "coming soon"
    if scheme == "Automatic via PCA":
        coef_fig = importance_plotter(X_train.columns,gs.best_estimator_._final_estimator.feature_importances_)

    else:
        features = indicatorMeta[indicatorMeta["Indicator Code"].isin(X_train.columns)]
        features["names"] =  features["Indicator"] + " (" + features["Dimension"] + ")"
        coef_fig = importance_plotter(features["names"],gs.best_estimator_._final_estimator.feature_importances_)


    units =  indicatorMeta[indicatorMeta["Indicator Code"]==target].Units.values[0]

    title = "Predictions with " + interval +" intervals ("+ indicatorMeta[indicatorMeta["Indicator Code"]==target].Units.values[0]+")"
    split_text = wrap(title)

    # Plot prediction with confidence intervals
    avp_fig = px.scatter(prediction, x="prediction" , y="country",error_x="upper",error_x_minus="lower",
        title='<br>'.join(split_text),color_continuous_scale=color_continuous_scale,labels={"color":units})
    avp_fig.update_layout(paper_bgcolor='#f2f2f2',plot_bgcolor='#f2f2f2')


    ### Bar plot with all the SIDS Countries
    bar_fig_list = bar_plotter(X_train,X_test,y_train,prediction.prediction,SIDS)

    # Produce alerts
    value_for_si = y_train.mean()
    SI_index = rmse/value_for_si
    # SI_index_msg = f"The model's  normalized root mean square is: {SI_index:.2f}."#
    t1 = time.time()
    exec_time = t1 - t0
    alert_msg = f"Trained and predicted in: {exec_time:.2f}s."

    return alert_msg,SI_index, avp_fig, coef_fig,query_card,bar_fig_list[0],bar_fig_list[1],bar_fig_list[2]
