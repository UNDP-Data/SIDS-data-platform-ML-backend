# This file contains all the hard coded values, These might need to move in to a database to improve extendability of
# the system.
import os

SIDS = ['ASM', 'AIA', 'ATG', 'ABW', 'BHS', 'BRB', 'BLZ', 'BES', 'VGB', 'CPV', 'COM', 'COK', 'CUB', 'CUW', 'DMA', 'DOM',
        'FJI', 'PYF',
        'GRD', 'GUM', 'GNB', 'GUY', 'HTI', 'JAM', 'KIR', 'MDV', 'MHL', 'MUS', 'FSM', 'MSR', 'NRU', 'NCL', 'NIU', 'MNP',
        'PLW', 'PNG', 'PRI',
        'KNA', 'LCA', 'VCT', 'WSM', 'STP', 'SYC', 'SGP', 'SXM', 'SLB', 'SUR', 'TLS', 'TON', 'TTO', 'TUV', 'VIR', 'VUT']

DATASETS_PATH = os.getenv("DATASET_PATH", "./datasets/")
