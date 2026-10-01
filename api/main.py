
from fastapi import FastAPI
from api.routes.predict import router as predict_router

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using a Random Forest model",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


app.include_router(predict_router)
