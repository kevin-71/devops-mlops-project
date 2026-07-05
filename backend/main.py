from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from backend.climate.service import ClimateService

app = FastAPI(title="Climate ML API", version="1.0.0")
Instrumentator().instrument(app).expose(app)
service = ClimateService()


class TrainRequest(BaseModel):
    refresh: bool = Field(default=False, description="Retrain even if artifacts already exist")


class ForecastRequest(BaseModel):
    months_ahead: int = Field(default=300, ge=1, le=300)
    model_name: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/status")
def status() -> dict:
    return service.get_status()


@app.post("/train")
def train(request: TrainRequest) -> dict:
    return service.train(force=request.refresh)


@app.get("/metrics")
def metrics() -> dict:
    current = service.load_or_train()
    return {
        "best_model_name": current["best_model_name"],
        "metrics": current["metrics"],
        "available_models": list(current["models"].keys()),
    }


@app.get("/forecast")
def forecast(months_ahead: int = 300, model_name: str | None = None) -> dict:
    try:
        return service.forecast(months_ahead=months_ahead, model_name=model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
