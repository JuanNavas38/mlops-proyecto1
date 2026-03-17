from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from botocore.client import Config
from botocore.exceptions import ClientError
import boto3
import pickle
import numpy as np
import os

# ── Mappings — deben ser idénticos a los del DAG de recolección ────────────
# La regla de oro: los datos de inferencia deben transformarse
# exactamente igual que los datos de entrenamiento.

WILDERNESS_AREAS = ['Rawah', 'Neota', 'Comanche Peak', 'Cache la Poudre']

SOIL_TYPES = [
    'Cathedral', 'Vanet', 'Haploborolis', 'Ratake', 'Vanet-Wetmore',
    'Vanet-Wetmore-Rock', 'Gothic', 'Supervisor', 'Troutville', 'Bullwark-Catamount',
    'Bullwark-Catamount-Rock', 'Legault', 'Catamount', 'Pachic-Argiborolis', 'Unspecified',
    'Cryaquolis-Cryoborolis', 'Gateview', 'Rogert', 'Typic-Cryaquolis', 'Typic-Cryaquepts',
    'Typic-Cryaquolls', 'Leighcan-till', 'Leighcan-till-Rock', 'Leighcan-stony',
    'Leighcan-Rock', 'Como-Legault', 'Family65-Leighcan', 'Catamount-Como',
    'Leighcan-Catamount', 'Leighcan-Catamount-Rock', 'Como-Rock-Leighcan',
    'Leighcan-Rock-Como', 'Cryorthents-Rock', 'Cryumbrepts-Rock', 'Bross-Rock',
    'Rock-Cryumbrepts', 'Leighcan-Moran', 'Moran-Cryorthents-Leighcan',
    'Moran-Cryorthents-Rock', 'Moran-Rock-Cryorthents'
]

# Nombres de los 7 tipos de cobertura forestal
COVER_TYPE_NAMES = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

# ── Configuración MinIO ─────────────────────────────────────────────────────
MINIO_CONFIG = {
    "endpoint_url": os.environ.get("MINIO_ENDPOINT", "http://minio:9000"),
    "aws_access_key_id": "minioadmin",
    "aws_secret_access_key": "minioadmin123",
    "config": Config(signature_version="s3v4")
}

BUCKET_NAME = "models"

# ── Carga del modelo al arrancar ────────────────────────────────────────────
model = None
model_name = None


def load_latest_model():
    """
    Lista todos los .pkl del bucket 'models' en MinIO,
    los ordena por nombre (que contiene timestamp) y carga el más reciente.
    """
    global model, model_name

    s3 = boto3.client("s3", **MINIO_CONFIG)

    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME)
        contents = response.get("Contents", [])

        pkl_files = sorted([obj["Key"] for obj in contents if obj["Key"].endswith(".pkl")])

        if not pkl_files:
            print("No hay modelos en MinIO todavía.")
            return False

        latest = pkl_files[-1]
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=latest)
        model = pickle.loads(obj["Body"].read())
        model_name = latest

        print(f"Modelo cargado: {model_name}")
        return True

    except ClientError as e:
        print(f"Error al conectar con MinIO: {e}")
        return False


# Intentar cargar el modelo al iniciar el contenedor
load_latest_model()

# ── Aplicación FastAPI ──────────────────────────────────────────────────────
app = FastAPI(
    title="Inference API - Forest Cover Type",
    description="Predice el tipo de cobertura forestal usando el modelo entrenado en MinIO.",
    version="1.0.0"
)


# ── Esquema de entrada ──────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    elevation: float
    aspect: float
    slope: float
    horizontal_distance_to_hydrology: float
    vertical_distance_to_hydrology: float
    horizontal_distance_to_roadways: float
    hillshade_9am: float
    hillshade_noon: float
    hillshade_3pm: float
    horizontal_distance_to_fire_points: float
    wilderness_area: str
    soil_type: str


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check — confirma que la API está viva y qué modelo tiene cargado."""
    return {
        "status": "ok",
        "model_loaded": model_name if model_name else "ninguno",
    }


@app.post("/predict")
def predict(request: PredictRequest):
    """
    Recibe los datos crudos de un punto del bosque y devuelve
    la predicción del tipo de cobertura forestal.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No hay modelo disponible. Ejecuta primero el DAG model_training en Airflow."
        )

    # ── 1. Validación (misma lógica que process_data en el DAG de recolección)
    hillshades = [request.hillshade_9am, request.hillshade_noon, request.hillshade_3pm]
    if not all(0 <= h <= 255 for h in hillshades):
        raise HTTPException(status_code=400, detail="Los valores de hillshade deben estar entre 0 y 255.")

    if request.wilderness_area not in WILDERNESS_AREAS:
        raise HTTPException(
            status_code=400,
            detail=f"wilderness_area '{request.wilderness_area}' no es válido. Opciones: {WILDERNESS_AREAS}"
        )

    if request.soil_type not in SOIL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"soil_type '{request.soil_type}' no es válido."
        )

    # ── 2. One-Hot Encoding (misma lógica que prepare_training_data en el DAG)
    wa = [0, 0, 0, 0]
    wa[WILDERNESS_AREAS.index(request.wilderness_area)] = 1

    st = [0] * 40
    st[SOIL_TYPES.index(request.soil_type)] = 1

    # ── 3. Construir el vector de 54 features en el mismo orden que training_data
    features = [
        request.elevation,
        request.aspect,
        request.slope,
        request.horizontal_distance_to_hydrology,
        request.vertical_distance_to_hydrology,
        request.horizontal_distance_to_roadways,
        request.hillshade_9am,
        request.hillshade_noon,
        request.hillshade_3pm,
        request.horizontal_distance_to_fire_points,
    ] + wa + st  # 10 + 4 + 40 = 54 features

    # ── 4. Predicción
    prediction = int(model.predict(np.array([features]))[0])

    return {
        "cover_type": prediction,
        "cover_type_name": COVER_TYPE_NAMES.get(prediction, "Desconocido"),
        "model_used": model_name
    }
