# This file contains all the hard coded values, These might need to move in to a database to improve extendability of
# the system.
import os

SIDS = ['ATG','ABW', 'BHS', 'BMU', 'BHR', 'BRB', 'BLZ', 'VGB', 'CPV', 'CYM', 'COM', 'CUB', 'CUW', 'DMA', 'DOM', 'FJI',
 'GRD', 'GNB', 'GUY', 'HTI', 'JAM', 'KIR', 'MDV', 'MHL', 'MUS', 'FSM', 'NRU', 'PLW', 'PNG', 'WSM', 'STP', 'SYC',
 'SGP', 'SXM', 'SLB', 'KNA', 'VCT', 'LCA', 'SUR', 'TLS', 'TTO', 'TON', 'TUV', 'TCA', 'VUT', 'AIA', 'COK', 'MSR',
 'TKL', 'NIU']

DATASETS_PATH = os.getenv("DATASET_PATH", "./datasets/")

MAIN_ENDPOINT_TAG = "Main Endpoint"
