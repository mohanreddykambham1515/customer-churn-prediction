
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.services.prediction_service import predict_churn

router = APIRouter()


class CustomerData(BaseModel):
    features: list[float]


@router.post("/predict")
def predict(customer: CustomerData):
    if len(customer.features) != 39:
        raise HTTPException(
            status_code=422,
            detail=f"Expected 39 features, received {len(customer.features)}."
        )

    try:
        return predict_churn(customer.features)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
