import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import manifold



import os
import logging


from common.constants import SIDS, DATASETS_PATH


def structure_data(data):
    """Restructure indicatorData into a (indicator x year) by country structure"""
    data = data.set_index(['Country Code', 'Indicator Code'])
    data = data.stack()
    data = data.unstack(['Country Code'])
    data = data.sort_index()
    return data

def restrcture(data,codes):
    """Subset indicatorData for indicator code in codes"""
    sub_data = data[data["Indicator Code"].isin(codes)]
    sub_data = structure_data(sub_data)
    return sub_data

def cou_ind_miss(Data):
    """
        Returns the amount of missingness across the years in each indicator-country pair
    """
    absolute_missing = Data.drop(columns=["Country Code", "Indicator Code"]).isnull().sum(axis=1)
    total = Data.drop(columns=["Country Code", "Indicator Code"]).count(axis=1)

    percent_missing = absolute_missing * 100 / total
    missing_value_df = pd.DataFrame({'row_name': Data["Country Code"] + "-" + Data["Indicator Code"],
                                     'Indicator Code': Data["Indicator Code"],
                                     'absolute_missing': absolute_missing,
                                     'total': total,
                                     'percent_missing': percent_missing})
    countyIndicator_missingness = missing_value_df.sort_values(["percent_missing", "row_name"])

    return countyIndicator_missingness


def data_importer(path=DATASETS_PATH):
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

    import os
    cwd = os.getcwd()
    logging.info("Current directory [%s]", cwd)
    logging.info('loading %s', path + "indicatorMeta.csv")

    try:
        indicatorMeta = pd.read_csv(path + "indicatorMeta.csv")
        logging.info('indicatorMeta.csv loaded %s', path + "indicatorMeta.csv")

        datasetMeta = pd.read_csv(path + "datasetMeta.csv")

        logging.info('datasetMeta.csv loaded')

        indicatorData = pd.read_csv(path + "indicatorData.csv")

        logging.info('indicatorData.csv loaded')
    except Exception as e:
        logging.exception("Read csv failed: " + str(e))

    #### Remove rows with missing country or indicator names
    indicatorData["Country/Indicator Code"] = indicatorData["Country Code"] + "-" + indicatorData["Indicator Code"]
    indicatorData = indicatorData[indicatorData["Country/Indicator Code"].notna()].drop(
        columns="Country/Indicator Code")

    indicatorMeta = indicatorMeta[indicatorMeta["Indicator Code"].isin(indicatorData["Indicator Code"].values)]
    indicatorMeta = indicatorMeta[indicatorMeta.Indicator.notna()]

    datasetMeta = datasetMeta[datasetMeta["Dataset Code"].isin(indicatorMeta["Dataset"].values)]
    datasetMeta = datasetMeta[datasetMeta["Dataset Name"].notna()]

    return indicatorMeta, datasetMeta, indicatorData

def kmeans_missing(X, n_clusters, max_iter=100):
    """Perform K-Means clustering on data with missing values.

    Args:
      X: An [n_samples, n_features] array of data to cluster.
      n_clusters: maximum Number of clusters to form.
      max_iter: Maximum number of EM iterations to perform.
    
    Bug: Iteratively imputing using the same algorithm might unrealistically reduce intracluster variation
        for large amount of missing data

    Returns:
      labels: An [n_samples] vector of integer labels.
      centroids: An [n_clusters, n_features] array of cluster centroids.
      X_hat: Copy of X with the missing values filled in.
    """

    # Initialize missing values to their column means
    missing = ~np.isfinite(X)
    mu = np.nanmean(X, 0, keepdims=1)
    X_hat = np.where(missing, mu, X)
    

    for i in range(max_iter+100):
        if i > 0:
            # initialize KMeans with the previous set of centroids. this is much
            # faster and makes it easier to check convergence (since labels
            # won't be permuted on every iteration), but might be more prone to
            # getting stuck in local minima.
            cls = KMeans(n_clusters, init=prev_centroids)
        else:
            # do multiple random initializations in parallel
            cls = KMeans(n_clusters)

        # perform clustering on the filled-in data
        labels = cls.fit_predict(X_hat)
        centroids = cls.cluster_centers_

        # fill in the missing values based on their cluster centroids
        X_hat[missing] = centroids[labels][missing]

        # when the labels have stopped changing then we have converged
        if i > 0 and np.all(labels == prev_labels):
            break

        prev_labels = labels
        prev_centroids = cls.cluster_centers_

    return labels, centroids, X_hat


# Import data
indicatorMeta = None
datasetMeta = None
indicatorData = None


def load_dataset():
    global indicatorMeta, datasetMeta, indicatorData
    if indicatorData is None:
        indicatorMeta, datasetMeta, indicatorData = data_importer()


if os.getenv("MODEL_SERVICE") is None or os.getenv("MODEL_SERVICE") == "twolvlImp":
    load_dataset()

# Sub-select SIDS 
indicatorData = indicatorData[indicatorData["Country Code"].isin(SIDS)]


def correlation_function(dataset,category,country,year):
    year = str(year)
    codes = np.unique(indicatorMeta[(indicatorMeta.Dataset==dataset)&(indicatorMeta.Category==category)]["Indicator Code"].values)
    data = restrcture(indicatorData[["Country Code","Indicator Code",year]],codes)

    data.dropna(how='all',inplace=True)
    corr = data.corr()
    corr = corr.fillna('')
    logging.info(corr.columns)
    country_corr=corr[[country]].drop(index=country).sort_values(by=country, ascending=False).reset_index()#.rename(columns={"index":"country"})

    return country_corr.to_dict(orient='list')

def cluster_function(dataset,category,year,k):
    year = str(year)
    codes = np.unique(indicatorMeta[(indicatorMeta.Dataset==dataset)&(indicatorMeta.Category==category)]["Indicator Code"].values)
    data = restrcture(indicatorData[["Country Code","Indicator Code",year]],codes)
    data = data.transpose()
    data.dropna(how='all',inplace=True)
    k = min(k, data.shape[0])
    labels, centroids, data_imputed = kmeans_missing(data, n_clusters=k)

    return data.index.tolist(),labels.tolist()