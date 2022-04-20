import logging

import pandas as pd

from common.constants import DATASETS_PATH

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


def data_importer(percent=90, model_type="non-series", path=DATASETS_PATH):
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

    if model_type == "series":
        # Indicators measured less than 5 times for each country are removed
        countyIndicator_missingness = cou_ind_miss(indicatorData)
        indicator_subset = set(countyIndicator_missingness[countyIndicator_missingness.percent_missing >= percent][
                                   "Indicator Code"]) - set(
            countyIndicator_missingness[countyIndicator_missingness.percent_missing < percent]["Indicator Code"])

        indicatorData = indicatorData[~indicatorData["Indicator Code"].isin(indicator_subset)]


    indicatorMeta = indicatorMeta[indicatorMeta["Indicator Code"].isin(indicatorData["Indicator Code"].values)]
    indicatorMeta = indicatorMeta[indicatorMeta.Indicator.notna()]

    datasetMeta = datasetMeta[datasetMeta["Dataset Code"].isin(indicatorMeta["Dataset"].values)]
    datasetMeta = datasetMeta[datasetMeta["Dataset Name"].notna()]

    return indicatorMeta, datasetMeta, indicatorData