from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import joblib
import logging
import os
import time

import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/api.log")
    ]
)

logger = logging.getLogger(__name__)

MODEL_VERSION = "1.0.0"

# Ordre des colonnes identique à celui de l'entraînement (src/train.py)
FEATURES = [
    "age",
    "anciennete",
    "revenu",
    "nb_reclamations",
    "montant_mensuel",
    "retards_paiement"
]

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
        # Correction 1 : DataFrame avec les noms de colonnes de l'entraînement
        values = pd.DataFrame([data.model_dump()])[FEATURES]

        # Correction 2 : un seul passage dans le modèle au lieu de deux
        probas = model.predict_proba(values)[0]
        prediction = int(model.classes_[probas.argmax()])
        probability = float(probas[1])

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
        "probability": round(probability, 4),
        "model_version": MODEL_VERSION,
        "duration_ms": round(duration * 1000, 2)
    }


Instrumentator().instrument(app).expose(app)
