import logging
import sys

logger = logging.getLogger('azure.mgmt.resource')
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)