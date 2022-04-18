import pandas as pd
import numpy as np
## for machine learning
from sklearn import preprocessing, cluster
import scipy
from scipy import cluster as scicluster

#!pip install kmodes
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes





def check_coordinate(data,metadata):
	"""
		using metadata check if there are observations in data without GPS coordinate
	"""

	gps_long = metadata["GPS longitude" in metadata.Type]["Question Name"].values[0]
	gps_lat = metadata["GPS latitude" in metadata.Type]["Question Name"].values[0]

	# Remove observations with not GPS measurements
	data = data[(data[gps_long].notna())&(data[gps_lat].notna())]

	data.dropna(how='all', inplace=True)

	return data



############## CLUSTERING
def optimal_cluster_cat(data, features, max_k,prefix):
    """Cluster data obersvations using kmodes clustering algorithm according to list of variables in features
    Args:
        data: dataset
        features: variables to be used for clustering
        max_k: maximum number of cluster to consider
        prefix: prefix for the cluster variable label feature to be added to the dataset
    Returns:
        data: orginial dataset with cluster labels added as features
    
    """
    X = data[features].copy()
    X.fillna(value="unknown",inplace=True)

    ####### elbow method ########
    ## iterations
    distortions = [] 
    for i in range(1, max_k+1):
        if len(X) >= i:
            kmode = KModes(n_clusters=i, init = "random", n_init = 10, verbose=0)
            kmode.fit_predict(X)
            distortions.append(kmode.cost_)
    ## best k: the lowest derivative
    k = [i*100 for i in np.diff(distortions,2)].index(min([i*100 for i in np.diff(distortions,2)]))

    ####### Actual clustering ########
    model = KModes(n_clusters=k, init = "random", n_init = 5, verbose=0)
    ## clustering
    dtf_X = X.copy()
    dtf_X["cluster"] = model.fit_predict(X)

    ## add clustering info to the original dataset
    data[[prefix+"_cluster"]] = dtf_X[["cluster"]]
    return data

def optimal_cluster_int(data, features, max_k,prefix):
    """Cluster data obersvations using kmeans clustering algorithm according to list of variables in features
    Args:
        data: dataset
        features: variables to be used for clustering
        max_k: maximum number of cluster to consider
        prefix: prefix for the cluster variable label feature to be added to the dataset
    Returns:
        data: orginial dataset with cluster labels added as features
    
    """
    X = data[features]
    
    ####### elbow method ########
    ## iterations
    distortions = [] 
    for i in range(1, max_k+1):
        if len(X) >= i:
            model = cluster.KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            model.fit(X)
            distortions.append(model.inertia_)
    ## best k: the lowest derivative
    k = [i*100 for i in np.diff(distortions,2)].index(min([i*100 for i 
         in np.diff(distortions,2)]))
    
    ####### Actual clustering ########
    model = cluster.KMeans(n_clusters=k, init='k-means++')
    ## clustering
    dtf_X = X.copy()
    dtf_X["cluster"] = model.fit_predict(X)
    ## find real centroids
    closest, distances = scicluster.vq.vq(model.cluster_centers_, 
                         dtf_X.drop("cluster", axis=1).values)
    dtf_X["centroids"] = 0
    for i in closest:
        dtf_X["centroids"].iloc[i] = 1
    ## add clustering info to the original dataset
    data[[prefix+"_cluster",prefix+"_centroids"]] = dtf_X[["cluster","centroids"]]
    return data

def optimal_cluster_mixed(data, features,cat_cols, max_k,prefix):
    """Cluster data obersvations using k-protoypes clustering algorithm according to list both cateogircal and numeric
    variables
    Args:
        data: dataset
        features: variables to be used for clustering
        cat_cols: cateogrical columns in features
        max_k: maximum number of cluster to consider
        prefix: prefix for the cluster variable label feature to be added to the dataset
    Returns:
        data: orginial dataset with cluster labels added as features
    
    """
    
    X = data[features].copy()
    
    catColumnsPos = [X.columns.get_loc(col) for col in cat_cols]
    X[cat_cols]= X[cat_cols].fillna(value="missing")
    print(X.head())
    dfMatrix = X.to_numpy()
    ####### elbow method ########

    # Choose optimal K using Elbow method
    cost = []
    for cluster in range(1, max_k+1):
        kprototype = KPrototypes(n_jobs = -1, n_clusters = cluster, init = 'Huang', random_state = 0)
        kprototype.fit_predict(dfMatrix, categorical = catColumnsPos)
        cost.append(kprototype.cost_)
        print('Cluster initiation: {}'.format(cluster))

    print(cost)

    k = [i*100 for i in np.diff(cost,2)].index(min([i*100 for i in np.diff(cost,2)]))
    
    
    # Fit the cluster
    kprototype = KPrototypes(n_jobs = -1, n_clusters = k, init = 'Huang', random_state = 0)
    kprototype.fit_predict(dfMatrix, categorical = catColumnsPos)
    data[[prefix+"_cluster"]] = kprototype.labels_
    return data



################ Function that returns a reponse to user/frontend ###########
def query_and_train(data,metadata,factors, max_k):
    data = pd.DataFrame.from_dict(data)

    metadata = pd.DataFrame.from_dict(metadata)

    data = check_coordinate(data,metadata)

    # Run clustering  based on the variable types included in this factor
    types = metadata[(metadata.Factor==factors)].Type.unique().tolist()

    assert (("categorical" in types)|('continuous' in types)), f"No categorical or continuous variables for clustering"


    if ("categorical" in types) & ('continuous' in types):
        cont_cols = metadata[(metadata.Factor==factors)&(metadata.Type=='continuous')]["Question Name"]
        cat_cols = metadata[(metadata.Factor==factors)&(metadata.Type=="categorical")]["Question Name"]
        data = optimal_cluster_mixed(data,cat_cols+cont_cols ,cat_cols, max_k,factors)
    elif ("categorical" in types):
        features = metadata[(metadata.Factor==factors)&(metadata.Type=="categorical")]["Question Name"]
        data = optimal_cluster_cat(data, features, max_k,factors)
    else:
        features = metadata[(metadata.Factor==factors)&(metadata.Type=='continuous')]["Question Name"]
        data = optimal_cluster_int(data, features, max_k,factors)
    return data[[factors+"_cluster"]].to_dict(orient='list')















