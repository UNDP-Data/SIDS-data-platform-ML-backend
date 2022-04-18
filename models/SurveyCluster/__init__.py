from typing import List, Optional, Dict
from typing import Optional

from fastapi import APIRouter
from pydantic import Field

from common.base_definition import BaseDefinition
from models.SurveyCluster.model import query_and_train


class ClusterRequest(BaseDefinition):

    data: Optional[Dict] = Field(None, title="data json")
    metadata: Optional[Dict] = Field(None, title="metadata json")
    factor: str = Field(..., title="factor to be clustered")
    max_k: int = Field(..., title="Maximum number of clusters expected", example=10)

class ClusterResponse(BaseDefinition):
    cluster_label: Optional[List[float]] = Field(..., description="cluster label for each observation")


router = APIRouter(
    prefix="/survey",
    tags=["Survey Analysis"],
    responses={404: {"description": "Not found"}},
)


@router.post('/clustering', response_model=ClusterResponse)
async def cluster(req: ClusterRequest):
    cluster_label = query_and_train(req.data,req.metadata,req.factor,req.max_k)

    return ClusterResponse(cluster_label=cluster_label)


