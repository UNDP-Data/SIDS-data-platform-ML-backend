import os

from common.logger import logger


class DataLoader(object):
    def __init__(self):
        self.store = {}

    def load_data(self, service_name: str, load_func):
        key = service_name
        if os.getenv("MODEL_SERVICE") is None or os.getenv("MODEL_SERVICE") == service_name:
            if key not in self.store:
                logger.info("Data loading for key %s", key)
                vargs = load_func
                self.store[key] = vargs
            else:
                logger.info("Data loaded from the cache %s", key)

            return self.store[key]


data_loader = DataLoader()
