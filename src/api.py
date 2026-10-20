from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from pathlib import Path
import pandas as pd
import joblib
import logging
import time

MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "model.joblib"
bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
feature_columns = bundle["feature_columns"]
model_name = bundle["model_name"]

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Educational demo only; not for medical diagnosis.",
    version="1.0.0"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart-disease-api")
request_count = 0

class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float | None = None
    thal: float | None = None

@app.middleware("http")
async def request_logging(request, call_next):
    global request_count
    started = time.perf_counter()
    response = await call_next(request)
    request_count += 1
    elapsed = (time.perf_counter() - started) * 1000
    logger.info(
        "method=%s path=%s status=%s duration_ms=%.2f",
        request.method, request.url.path, response.status_code, elapsed
    )
    response.headers["X-Process-Time-ms"] = f"{elapsed:.2f}"
    return response

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}

@app.get("/metrics")
def metrics():
    return {"api_requests_total": request_count}

@app.post("/predict")
def predict(patient: PatientFeatures):
    patient_df = pd.DataFrame([patient.model_dump()])
    patient_df = patient_df.reindex(columns=feature_columns)
    try:
        prediction = int(pipeline.predict(patient_df)[0])
        probabilities = pipeline.predict_proba(patient_df)[0]
        positive_index = list(pipeline.classes_).index(1)
        return {
            "prediction": prediction,
            "label": "disease_present" if prediction == 1 else "disease_absent",
            "positive_class_probability": round(float(probabilities[positive_index]), 4),
            "model": model_name,
            "disclaimer": "Educational model output only; not a diagnosis."
        }
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=422, detail="Input could not be processed.") from exc
