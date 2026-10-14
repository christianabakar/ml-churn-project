from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import joblib
import logging
import time

from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger(__name__)

MODEL_VERSION = "1.0.0"

app = FastAPI(
    title="Churn Prediction API",
    version=MODEL_VERSION
)

model = joblib.load("models/model.pkl")

logger.info(f"event=model_loaded model=RandomForest model_version={MODEL_VERSION}")


class ClientData(BaseModel):
    age: int
    anciennete: int
    revenu: float
    nb_reclamations: int
    montant_mensuel: float
    retards_paiement: int


@app.get("/")
def root():
    return {
        "application": "Churn Prediction API",
        "version": MODEL_VERSION,
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "UP"
    }


@app.get("/model/info")
def model_info():
    return {
        "model": "Random Forest",
        "version": MODEL_VERSION
    }


@app.post("/predict")
def predict(data: ClientData, request: Request):

    start = time.time()
    client_ip = request.client.host if request.client else "unknown"

    logger.info(
        f"event=prediction_request client_ip={client_ip} "
        f"model_version={MODEL_VERSION}"
    )

    try:
        values = [[
            data.age,
            data.anciennete,
            data.revenu,
            data.nb_reclamations,
            data.montant_mensuel,
            data.retards_paiement
        ]]

        prediction = model.predict(values)[0]
        probability = model.predict_proba(values)[0][1]

    except Exception as e:
        duration = time.time() - start
        logger.error(
            f"event=prediction_error client_ip={client_ip} "
            f"model_version={MODEL_VERSION} "
            f"duration_ms={duration * 1000:.2f} error=\"{e}\""
        )
        raise HTTPException(status_code=500, detail="Prediction error")

    duration = time.time() - start
    label = "CHURN" if prediction == 1 else "NO_CHURN"

    logger.info(
        f"event=prediction_success client_ip={client_ip} "
        f"model_version={MODEL_VERSION} prediction={label} "
        f"probability={probability:.4f} duration_ms={duration * 1000:.2f}"
    )

    return {
        "prediction": label,
        "probability": round(float(probability), 4),
        "model_version": MODEL_VERSION,
        "duration_ms": round(duration * 1000, 2)
    }


Instrumentator().instrument(app).expose(app)
