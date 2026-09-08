from fastapi import FastAPI
from src.Backend.Api.routes import router

app = FastAPI(
    title="Dr. ML Prediction App", 
    version="1.0.0",
    description="Multi-disease prediction backend"

    )

app.include_router(router, prefix="/api")