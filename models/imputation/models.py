# For data manipulation


# For plotting
import pandas as pd
import numpy as np
import pycountry
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
# from sklearn
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline

# For helper functions
from common.constants import SIDS

#import plotly.graph_objs as go
from common.utils import data_importer

wb_data,indicatorMeta, datasetMeta,indicatorData = data_importer(model_type="knn")


def model_trainer(X_train,X_test,y_train,seed,n_estimators,t0,target_year,target, model,interval):
	"""
    Train the selected model, cross validate, score and generate a 90% prediction interval based on bootstrapped residuals.
    Args:
        X_train: training data
		X_test: prediction data
		y_train: training target array
		seed: random state setter
		n_estimators: number of trees for tree based models,
		t0: timer count since preprocessing
        predict_year: the year that we are targetting
        target: name feature to be used as target
        model: type of model to be trained
    Returns:
		alert: model training time alert
		SI_alert: model performance alert
		avp_fig: prediction plot with confidence intervals
		coef_fig: feature importance plot
		bar_fig_list: barplot for target varaibles by SIDS country location
	"""

	if model == "all":
		model = ["rfr","etr","gbr","svr"]
	model_instances=[]
	params= []

	num_folds = 5
	scoring = 'neg_mean_squared_error'
	if "rfr" in  model:
		clf1 = RandomForestRegressor(random_state = seed)
		param1 = {}
		param1['regressor__n_estimators'] = [n_estimators]#[10, 50, 100]
		param1['regressor__max_depth'] = [5, 10, 20,100, None]
		param1['regressor'] = [clf1]
		model_instances.append(clf1)
		params.append(param1)
	if "etr" in model:
		clf2 = ExtraTreesRegressor(random_state = seed)
		param2 = {}
		param2['regressor__n_estimators'] = [n_estimators]#[10, 50, 100]
		param2['regressor__max_depth'] = [5, 10, 20,100, None]
		param2['regressor'] = [clf2]	
		model_instances.append(clf2)
		params.append(param2)

	if "gbr" in model:
		clf3 = GradientBoostingRegressor(random_state = seed)
		param3 = {}
		if interval == "quantile":
			param3['regressor__loss'] = ['quantile']
			param3['regressor__alpha'] = [0.5]
		param3['regressor__n_estimators'] = [n_estimators]#[10, 50, 100]
		param3['regressor__max_depth'] = [3,5, 10, 20, None]
		param3['regressor'] = [clf3]	
		model_instances.append(clf3)
		params.append(param3)
	#if "svr" in model:
	#	clf4 = SVR()
	#	param4 = {}
		#param4['regressor__kernel'] = ['linear', 'poly', 'rbf']#[10, 50, 100]
		#param4['regressor__degree'] = [1,2,3]
	#	param4['regressor'] = [clf4]	
	#	model_instances.append(clf4)
	#	params.append(param4)

	pipeline = Pipeline([('regressor', model_instances[0])])
	gs = GridSearchCV(pipeline, params, cv=num_folds, n_jobs=-1, scoring=scoring, refit=True).fit(X_train, y_train)
	rmse = np.sqrt(-gs.best_score_)

	best_model = gs.best_estimator_["regressor"]
	
	prediction = pd.DataFrame(gs.predict(X_test), columns=["prediction"], index=X_test.index)


	if interval == "bootstrap":
	
		#Residual Bootsrapping  on validation data
		pred_train = cross_val_predict(best_model,X_train, y_train, cv=3)

		res = y_train - pred_train

		### BOOTSTRAPPED INTERVALS ###

		alpha = 0.1 #(90% prediction interval)

		bootstrap = np.asarray([np.random.choice(res, size=res.shape) for _ in range(100)])
		q_bootstrap = np.quantile(bootstrap, q=[alpha/2, 1-alpha/2], axis=0)

		#prediction = pd.DataFrame(gs.predict(X_test), columns=["prediction"], index=X_test.index)
		prediction["upper"]= prediction["prediction"] + q_bootstrap[1].mean()
		prediction["lower"]= prediction["prediction"] + q_bootstrap[0].mean()
	
	else:
		if str(type(best_model))== "<class 'sklearn.ensemble._gb.GradientBoostingRegressor'>":
			all_models = {}
			for alpha in [0.05, 0.95]:
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
			quantiles = [0.05, 0.95]
			
			for q in quantiles:
				s = pred_Q.quantile(q=q, axis=1)
				prediction[str(q)] = s.values
			prediction.rename(columns={"0.05":"lower","0.95":"upper"}, inplace=True)

	# Predict for SIDS countries with missing values
	prediction = prediction[prediction.index.isin(SIDS)]
	prediction.index = [pycountry.countries.get(alpha_3=i).name for i in prediction.index]
	prediction = prediction.reset_index().rename(columns={"index":"country"})

	#################### Prediction dataframe and best_model instance are the final results of the ML################

	return prediction,rmse,gs, best_model




# def random_forest(X_train,X_test,y_train,y_test,seed,n_estimators,t0,target_year,target):

# 	"""Function for training a random forest regressor model"""

# 	print("Basic Modelling")
# 	seed = 7
# 	num_folds = 5
# 	scoring = 'neg_mean_squared_error'

# 	clf2 = RandomForestRegressor(random_state = seed,n_estimators=n_estimators, max_features="sqrt")

# 	#For SI INDEX
# 	param2 = {}
# 	param2['regressor__n_estimators'] = [n_estimators]
# 	param2['regressor'] = [clf2]
# 	pipeline = Pipeline([('regressor', clf2)])
# 	params = [param2]
# 	gs = GridSearchCV(pipeline, params, cv=num_folds, n_jobs=-1, scoring=scoring).fit(X_train, y_train)
# 	rmse = np.sqrt(-gs.best_score_)
# 	value_for_si = y_train.mean()
# 	SI_index = rmse/value_for_si
# 	SI_index_msg = f"The model's normalized root mean square is: {SI_index:.2f}."

# 	#For prediction 
# 	clf2.fit(X_train,y_train)

# 	# Calculate the variance/confidence interval
# 	predictions = clf2.predict(X_test)
# 	V_IJ_unbiased = fci.random_forest_error(clf2, X_train, X_test)



# 	#X_test.reset_index(inplace = True)

# 	# Predict for SIDS countries with missing values
# 	countries = X_test.index
# 	countries = list(set(countries).intersection(set(SIDS)))
# 	countries = [pycountry.countries.get(alpha_3=i).name for i in countries]
# 	prediction = pd.DataFrame(data={"prediction": predictions,"ci": np.sqrt(V_IJ_unbiased), "country":countries})


# 	# Plot feature importance
	
# 	features = indicatorMeta[indicatorMeta["Indicator Code"].isin(X_train.columns)]
# 	features["names"] =  features["Indicator"] + " (" + features["Dimension"] + ")"

# 	coef_fig = importance_plotter(features["names"],clf2.feature_importances_)



# 	# Plot prediction with confidence intervals
# 	units =  indicatorMeta[indicatorMeta["Indicator Code"]==target].Units.values[0]

# 	title = "Predictions with Infinite Jacknife intervals ("+ indicatorMeta[indicatorMeta["Indicator Code"]==target].Units.values[0]+")"
# 	split_text = wrap(title)
# 	avp_fig = px.scatter(prediction, x="prediction" , y="country", error_x="ci",
# 		title='<br>'.join(split_text), color="prediction",color_continuous_scale=color_continuous_scale,labels={"color":units})
# 	avp_fig.update_layout(paper_bgcolor='#f2f2f2',plot_bgcolor='#f2f2f2')


# 	### Bar plot with all the SIDS Countries
# 	bar_fig_list = bar_plotter(X_train,X_test,y_train,predictions,SIDS)



# 	# Produce alerts
# 	t1 = time.time()
# 	exec_time = t1 - t0
# 	alert_msg = f"Trained and predicted in: {exec_time:.2f}s."
# 	alert = dbc.Alert(alert_msg, color="success", dismissable=True)
# 	SI_alert = dbc.Alert(SI_index_msg, color="success", dismissable=True)
# 	#return clf2, prediction, viz

# 	return alert,SI_alert, avp_fig, coef_fig,bar_fig_list[0],bar_fig_list[1],bar_fig_list[2]






# def boost_regressor(X_train,X_test,y_train,y_test,seed,n_estimators,t0,target_year,target):

# 	"""Train a gradient boost regressor model"""

# 	# train the quantile gradient boost regression for a 95% prediction interval
# 	all_models = {}
# 	common_params = dict(
# 		learning_rate=0.05,
# 		n_estimators=n_estimators,

# 	)
# 	for alpha in [0.05, 0.5, 0.95]:
# 		if alpha == 0.5:
# 			clf2 = GradientBoostingRegressor(loss="quantile", alpha=alpha,max_features="sqrt", **common_params)
# 			score = gbr.train_score_[-1]
# 			all_models["q %1.2f" % alpha] = clf2.fit(X_train, y_train)
# 		else: 
# 			gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha,max_features="sqrt", **common_params)
# 			all_models["q %1.2f" % alpha] = gbr.fit(X_train, y_train)

# 	#For score
# 	print(score)
# 	SI_index_msg = f"The model's deviance score on training data  is: {score:.2f}."
# 	#SI_index_msg = f"The model's Score is: 2."


# 	#For prediction
# 	y_lower = all_models["q 0.05"].predict(X_test)
# 	y_upper = all_models["q 0.95"].predict(X_test)
# 	predictions = all_models["q 0.50"].predict(X_test)


# 	#X_test.reset_index(inplace = True)

# 	# Predict for SIDS countries with missing values
# 	countries = X_test.index
# 	countries = list(set(countries).intersection(set(SIDS)))
# 	countries = [pycountry.countries.get(alpha_3=i).name for i in countries]
# 	prediction = pd.DataFrame(data={"prediction": predictions,"lower": y_lower,"upper":y_upper, "country":countries})


# 	# Plot feature importance
# 	features = indicatorMeta[indicatorMeta["Indicator Code"].isin(X_train.columns)]
# 	features["names"] =  features["Indicator"] + " (" + features["Dimension"] + ")"
# 	coef_fig = importance_plotter(features["names"],clf2.feature_importances_)



# 	# Plot prediction with confidence intervals
# 	units =  indicatorMeta[indicatorMeta["Indicator Code"]==target].Units.values[0]

# 	title = "Predictions with Infinite Jacknife intervals ("+ indicatorMeta[indicatorMeta["Indicator Code"]==target].Units.values[0]+")"
# 	split_text = wrap(title)

# 	avp_fig = px.scatter(prediction, x="prediction" , y="country",error_x="upper",error_x_minus="lower"
#         ,title='<br>'.join(split_text),color_continuous_scale=color_continuous_scale,labels={"color":units})
# 	avp_fig.update_layout(paper_bgcolor='#f2f2f2',plot_bgcolor='#f2f2f2')

# 	### Bar plot with all the SIDS Countries
# 	bar_fig_list = bar_plotter(X_train,X_test,y_train,predictions,SIDS)



# 	# Produce alerts
# 	t1 = time.time()
# 	exec_time = t1 - t0
# 	alert_msg = f"Trained and predicted in: {exec_time:.2f}s."
# 	alert = dbc.Alert(alert_msg, color="success", dismissable=True)
# 	SI_alert = dbc.Alert(SI_index_msg, color="success", dismissable=True)
# 	#return clf2, prediction, viz

# 	return alert,SI_alert, avp_fig, coef_fig,bar_fig_list[0],bar_fig_list[1],bar_fig_list[2]
