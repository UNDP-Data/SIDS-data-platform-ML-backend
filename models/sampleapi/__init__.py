from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field


class SampleRequest(BaseModel):
    requiredField: str = Field(None, title="This field is required", example="required 123")
    optionalField: Optional[str] = Field(None, title="This field is optional", example="optional 123")


class SampleResponse(BaseModel):
    resp1: str


router = APIRouter(
    prefix="/sample_model",
    tags=["Sample Model"],
    responses={404: {"description": "Not found"}},
)


@router.post('/test_endpoint1', response_model=SampleResponse)
async def test_endpoint1(req: SampleRequest):
    return SampleResponse(resp1="Test 1")


@router.post('/test_endpoint2')
async def test_endpoint1(name: str):
    return "Hi "+name
