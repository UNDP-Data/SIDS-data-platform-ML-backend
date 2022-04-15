from typing import Optional

from fastapi import APIRouter
from pydantic import Field

from common.base_definition import BaseDefinition


class SampleRequest(BaseDefinition):
    requiredField: str = Field(..., title="This field is required", example="required 123")
    optionalField: Optional[str] = Field(None, title="This field is optional", example="optional 123")


class SampleResponse(BaseDefinition):
    resp1: str


router = APIRouter(
    prefix="/sampleapi",
    tags=["Sample Model"],
    responses={404: {"description": "Not found"}},
)


@router.post('/predict', response_model=SampleResponse)
async def test_endpoint1(req: SampleRequest):
    return SampleResponse(resp1="Test 1")


@router.post('/test_endpoint2')
async def test_endpoint2(name: str):
    return "Hi "+name
