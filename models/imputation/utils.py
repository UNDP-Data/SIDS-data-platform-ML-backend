import pandas as pd
import numpy as np
import pycountry as pycountry
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import RFE
import plotly.express as px

color_continuous_scale=px.colors.sequential.Blues

def preprocessing(data,target, target_year,interpolator,percent=80):

    """
    Preprocess data into a format suitable for the model
        data: indicatorData dataset 
        target: indicator whose values will be imputed
        target_year: the year under consideration

    Returns: unimputed training data with columns with less than 80% np.nan
             filled dataset imputed using the selected interpolater
    """

    #Subset data for target and target_year
    wdi_indicatorData_2010 = data[["Country Code","Indicator Code",str(target_year)]]
    wdi_indicatorData_2010 = wdi_indicatorData_2010.set_index(["Country Code","Indicator Code"])[str(target_year)].unstack(level=1)

    #Find how much missing values are there for each indicator
    rank = missingness(wdi_indicatorData_2010)

    #Only consider indcators with less than 80% missing values
    top_ranked = rank[rank.percent_missing < percent]["column_name"].values

    training_data = wdi_indicatorData_2010[top_ranked]

    #interpolation for indcators missing less than 10% using KNN imputer
    most_complete = rank[rank.percent_missing < 30]["column_name"].values

    if interpolator == 'KNNImputer':
        scaler = MinMaxScaler()
        imputer = KNNImputer(n_neighbors=5)
        scaler.fit(training_data[most_complete])
        scaled_imputed = imputer.fit_transform(scaler.transform(training_data[most_complete]))

        filled_data = pd.DataFrame(data=scaler.inverse_transform(scaled_imputed),columns=training_data[most_complete].columns,
                              index=training_data[most_complete].index)

    elif interpolator == 'SimpleImputer':
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputed= imp_mean.fit_transform(X=training_data[most_complete])
        filled_data = pd.DataFrame(data=imputed,columns=training_data[most_complete].columns,
                              index=training_data[most_complete].index)


    else:
        imp = IterativeImputer(missing_values=np.nan,random_state=0, estimator=ExtraTreesRegressor(n_estimators=10, random_state=0),n_nearest_features=100, 
                         add_indicator=False,sample_posterior=False)
        imp.fit(training_data[most_complete])
        imputed = imp.transform(training_data[most_complete])
        filled_data = pd.DataFrame(data=imputed,columns=training_data[most_complete].columns,
                              index=training_data[most_complete].index)



    #filled_data = pd.DataFrame(data=imputer.fit_transform(training_data[most_complete]),columns=training_data[most_complete].columns,
    #                          index=training_data[most_complete].index)


    return training_data, filled_data

def correlation_plotter(target,features, data, meta):
    """
        Returns a simple correlation plot of features and target variable

    """
    if target in features:
        names = meta[meta["Indicator Code"].isin(features)].Indicator.values
        correlation = data[features].corr()
        correlation.index = names
        correlation.columns = names
    else:
        features_copy = features.copy()
        features_copy.append(target)
        names = meta[meta["Indicator Code"].isin(features_copy)].Indicator.values
        correlation = data[features_copy].corr()
        correlation.index = names
        correlation.columns = names

    fig = px.imshow(correlation, x=names, y =names,color_continuous_scale=color_continuous_scale, title= 'Correlation plot')
    return fig


def missingness(df):
    "Rank the columns of df by the amount of missing observations"
    absolute_missing = df.isnull().sum()
    percent_missing = absolute_missing * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                     'absolute_missing': absolute_missing,
                                     'percent_missing': percent_missing})
    return missing_value_df.sort_values(["percent_missing","column_name"])






def impute_data_interpolation(data, upto_year, method):
    """
    Interpolation of indicator time-series for each country seperately in indicator dataFrame 
    Args:
        data: multiIndex dataFrame of feature data with index of (country, year) and columns of the feature names.
        upto_year: Year up to and including to interpolate
        method: method of interpolation to be passed on to the pandas interpolation function
    Returns:
        interpolated dataFrame
    """
    
    #Protect data in caller 
    data_local = data.copy()
    
    #Interpolation. For loop could be removed if data restructured to have time-series on .. 
    # rows (i.e Country and indicator in Index and year as column) 
    for country in data.index.levels[0]:    
        data_local.loc[(country):(country,str(upto_year)),:] = \
            data_local.loc[(country):(country,str(upto_year)),:]. \
            interpolate( method=method, limit_direction='both').values  
    
    #Use mean of indicator for any remaining missing values   
    idx = pd.IndexSlice   
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(data_local.loc[idx[:,:str(upto_year)],:])
    transformed = imp_mean.transform(data_local.loc[idx[:,:str(upto_year)],:])
    data_local.loc[idx[:,:str(upto_year)],:] = transformed
        
    
    return data_local

def window_data(data, lag=5,num_windows=3, step=1, predict_year=2010, target=None, impute_type=None):
    """
    Split up input feature dataFrame into windowed data.
    Bug: Lags are wrong way around. For example if window width of 5 is specified (lag=5) then the lag=1 column..
        gives the 5th value while lag=5 gives the 1st value in the time window.
    Args:
        data: multiIndex dataframe of feature data with index of (country, year) and columns of the feature names.
        lag: size of window
        num_windows: number of windows to gererate
        step: the delta between the windows. 1 will mean that there is maximum overlap between windows.
        predict_year: the year that we are targetting
        target: feature to be used as target
        impute_type: must be one of ['interpolation'] or None
    Returns:
        data_regressors
        data_targets
    """
    assert(impute_type in ['interpolation', None]), "impute_type must be one of  ['interpolation'] or none"
    assert(target in list(data.columns.values)), "Target should be in the input dataframe"

    if impute_type == 'interpolation':
        impute_func = impute_data_interpolation
    else:
        impute_func = None

    countries_in_data = list(data.index.levels[0]) 
    idx = pd.IndexSlice

    #Create empty test and training dataframes
    regressors_index = pd.MultiIndex.from_product([countries_in_data,
                                                   list(range(1,num_windows+1)), 
                                                   list(range(1,lag+1))],
                                                  names=[u'country', u'window', u'lag'])

    target_index = pd.MultiIndex.from_product([countries_in_data,
                                               list(range(1,num_windows+1))],
                                              names=[u'country', u'window'])

    data_regressors = pd.DataFrame(index=regressors_index, columns=data.columns)
    data_targets = pd.DataFrame(index=target_index, columns=[target])


    #Each increment of window represents moving back a year in time
    for window in range(num_windows):
        year = predict_year - window
        print(year)
        #Redo Imputation every time we move back a year
        #This maintains the requirement not to use information from future years in our imputations 
        if impute_func is not None:
            data_imp = impute_func(data, upto_year=year-1, method='linear' )
        else:
            data_imp = data
        print("target index")
        print(data_targets.loc[idx[:,window+1],:].index)
        data_targets.loc[idx[:,window+1],:] = data_imp.loc[idx[:,str(year)], target].values

        print("Regressor index")
        print(data_regressors.loc[idx[:,window+1,1:lag+1],:].index)
        data_regressors.loc[idx[:,window+1,1:lag+1],:] = \
                data_imp.loc[idx[:,str(year-lag):str(year-1)], :].values

    #According to pandas docs on multiIndex usage: For objects to be indexed and sliced effectively, they need to be sorted.
    data_regressors = data_regressors.sort_index()
    data_targets = data_targets.sort_index()

    #unstacking the input features. Each row will now represent a set of features.
    data_regressors  = data_regressors.unstack(level=2)

    return data_regressors, data_targets



def tup_to_string(dataset):
    """
        Turns a list of tuples into a list of strings by joining the tuples
    """

    delimiter = ', '
    columns = [delimiter.join([str(value) for value in values]) for values in dataset.columns]
    indexes = [delimiter.join([str(value) for value in values]) for values in dataset.index]
    
    return pd.DataFrame(data=dataset.values, columns=columns, index=indexes)

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

def data_importer(model_type="series",path = "/Volumes/My Passport for Mac/jobs/UNDP/ML-IndicatorData/"):
    """
        Import csv files and restrructure the data into a country by indcator format. Model_type will be expanded upon.
    """
    #


    indicatorMeta = pd.read_csv(path + "indicatorMeta.csv")

    datasetMeta = pd.read_csv(path + "datasetMeta.csv")

    indicatorData = pd.read_csv(path + "indicatorData.csv")

    SIDS = ['ASM','AIA','ATG','ABW','BHS','BRB','BLZ','BES','VGB','CPV','COM','COK','CUB','CUW','DMA','DOM','FJI','PYF',
    'GRD','GUM','GNB','GUY','HTI','JAM','KIR','MDV','MHL','MUS','FSM','MSR','NRU','NCL','NIU','MNP','PLW','PNG','PRI',
    'KNA','LCA','VCT','WSM','STP','SYC','SGP','SXM','SLB','SUR','TLS','TON','TTO','TUV','VIR','VUT']


    #### Remove rows with missing country or indicator names
    indicatorData["Country/Indicator Code"] = indicatorData["Country Code"]+"-"+indicatorData["Indicator Code"]
    indicatorData= indicatorData[indicatorData["Country/Indicator Code"].notna()].drop(columns="Country/Indicator Code")

    if model_type == "series":
        # Indicators measured less than 5 times for each country are removed
        countyIndicator_missingness = cou_ind_miss(indicatorData)
        indicator_subset = set(countyIndicator_missingness[countyIndicator_missingness.percent_missing >= 90]["Indicator Code"])- set(countyIndicator_missingness[countyIndicator_missingness.percent_missing<90]["Indicator Code"])

        indicatorData=indicatorData[~indicatorData["Indicator Code"].isin(indicator_subset)]

    wb_data = indicatorData.set_index(['Country Code', 'Indicator Code'])
    wb_data = wb_data.stack()
    wb_data = wb_data.unstack(['Indicator Code'])
    wb_data = wb_data.sort_index()

    indicatorMeta=indicatorMeta[indicatorMeta["Indicator Code"].isin(indicatorData["Indicator Code"].values)]
    indicatorMeta=indicatorMeta[indicatorMeta.Indicator.notna()]

    datasetMeta=datasetMeta[datasetMeta["Dataset Code"].isin(indicatorMeta["Dataset"].values)]
    datasetMeta=datasetMeta[datasetMeta["Dataset Name"].notna()]

    return wb_data,indicatorMeta, datasetMeta, indicatorData


def locator(x):
    if x in ["Bahrain","Seychelles","Maldives","Mauritius","Singapore","Cabo Verde","Sao Tome and Principe",'Guinea-Bissau','Comoros']:
        return 'AIS'
    elif x in ["Bahamas",'Cuba','Turks and Caicos','Haiti','Dominican Republic','Virgin Islands, British' 'Anguilla',
               'Antigua and Barbuda','Aruba','Saint Kitts and Nevis',"Dominica",'Montserrat', 'Saint Lucia','Saint Vincent and the Grenadines',
              'Sint Maarten (Dutch part)','Barbados','Trinidad and Tobago','Grenada','Guyana','Suriname','Belize','Bermuda',
              'Jamaica','Cayman Islands','Curaçao','Puerto Rico']:
        return 'Caribbean'
    else:
        return 'Pacific'


def bar_plotter(X_train,X_test,y_train,predictions,SIDS):
    """Bar plot with all the SIDS Countries"""

    #Change iso_3 code to country name
    train_countries =  list(set(X_train.index).intersection(set(SIDS)))
    y_train = y_train[y_train.index.isin(train_countries)]
    train_countries = [pycountry.countries.get(alpha_3=i).name for i in train_countries]

    countries = X_test.index
    countries = list(set(countries).intersection(set(SIDS)))
    countries = [pycountry.countries.get(alpha_3=i).name for i in countries]

    #Create dataset with country name and value(observed and predicted)
    bar_dataset_x= pd.concat([pd.Series(train_countries, name="Country"),pd.Series(countries,name="Country")],axis=0)
    bar_dataset_y= pd.concat([pd.Series(y_train.values, name="value"),pd.Series(predictions, name="value")],axis=0)
    bar_dataset_z= pd.concat([pd.Series(["Observed"]*y_train.shape[0], name="Predicted or Observed"),pd.Series(["Predicted"]*len(countries), name="Predicted or Observed")])
    bar_dataset = pd.concat([bar_dataset_x,bar_dataset_y, bar_dataset_z],axis=1, join='inner')

    # Add location of the SIDS country
    bar_dataset["Country Location"] = bar_dataset["Country"].apply(lambda x:locator(x))

    # Split by location and plot separate bar charts
    bar_dataset_AIC = bar_dataset[bar_dataset['Country Location']=='AIS']
    bar_dataset_Caribbean= bar_dataset[bar_dataset['Country Location']=='Caribbean']
    bar_dataset_Pacific = bar_dataset[bar_dataset['Country Location']=='Pacific']


    datasets=[bar_dataset_AIC,bar_dataset_Caribbean,bar_dataset_Pacific]
    bar_fig_list=[]
    for i in datasets:
        bars=[]
        colors = {'Predicted': '#0D2A63',
                'Observed': '#1F77B4'}
        # for label, label_df in i.groupby('Predicted or Observed'):
        #     bars.append(go.Bar(y=label_df["Country"],
        #                         x=label_df["value"],
        #                         orientation= "h",
        #                         name=label,
        #                         marker={'color': colors[label]}))
        #
        # # bar_fig=go.FigureWidget(data=bars,layout = go.Layout(paper_bgcolor='#f2f2f2',plot_bgcolor='#f2f2f2'))
        # bar_fig_list.append(bar_fig)
    return bar_fig_list


def importance_plotter(features,feature_importances):
     # Create feature importance plot
    coef_fig_series = px.bar(
        x=features,
        y=feature_importances,
        orientation="v",
        color=feature_importances,
        labels={"color":"Importance","y": "Gini importance", "x": "Features"},
        title="Feature importance for imputing indicator",
        color_continuous_scale=color_continuous_scale
        
    )
    coef_fig_series.update_layout(paper_bgcolor='#f2f2f2',plot_bgcolor='#f2f2f2')
    return coef_fig_series

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