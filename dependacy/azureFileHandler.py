import logging
import os

from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential

class AzureFileHandler:

    def __init__(self):
        self.default_credential = DefaultAzureCredential()

    def list_files(self, directory):
        container = ContainerClient(account_url="https://sidsakstest.blob.core.windows.net", container_name=directory)
        blob_list = [ b.name for b in container.list_blobs()]
        logging.info(blob_list)
        return list(blob_list)
