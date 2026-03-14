from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from botocore.client import Config
from botocore.exceptions import ClientError
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import psycopg2
import pandas as pd
import pickle
import boto3
import json
import os

# ── Configuración ──────────────────────────────────────────────────────────

# Mínimo de filas para garantizar que se recolectaron los 10 batches
# 10 batches × ~5,810 filas por solicitud = ~58,100 filas
MIN_ROWS = 58000

PG_CONN = {
    "host": "postgres",
    "port": 5432,
    "dbname": "mlops_db",
    "user": "admin",
    "password": "admin123"
}

MINIO_CONFIG = {
    "endpoint_url": "http://minio:9000",
    "aws_access_key_id": "minioadmin",
    "aws_secret_access_key": "minioadmin123",
    "config": Config(signature_version="s3v4")
}

BUCKET_NAME = "models"

# ── Tareas ─────────────────────────────────────────────────────────────────

def check_data(**context):
    """
    Tarea 1: verifica que haya suficientes datos para entrenar.
    Falla si no se han recolectado los 10 batches completos.
    Esto evita entrenar con datos incompletos.
    """
    conn = psycopg2.connect(**PG_CONN)
    cur = conn.cursor()
    # Se valida sobre raw_data porque refleja exactamente lo que llegó de la API,
    # independientemente de cuántas filas se perdieron en la limpieza.
    cur.execute("SELECT COUNT(*) FROM raw_data")
    count = cur.fetchone()[0]
    cur.close()
    conn.close()

    print(f"Filas en raw_data (solicitudes recibidas de la API): {count}")

    if count < MIN_ROWS:
        raise Exception(
            f"Solicitudes insuficientes: {count} filas en raw_data. "
            f"Se necesitan al menos {MIN_ROWS} filas (10 batches completos). "
            f"Faltan aproximadamente {(MIN_ROWS - count) // 5810 + 1} ejecuciones del DAG de recolección."
        )

    print(f"Validación exitosa: {count} filas en raw_data — los 10 batches fueron recolectados.")


def train_model(**context):
    """
    Tarea 2: lee todos los datos de training_data, entrena un
    RandomForestClassifier y guarda el modelo en un archivo temporal.
    Pasa la ruta del archivo y las métricas a la siguiente tarea via XCom.
    """
    # Leer todos los datos de training_data
    conn = psycopg2.connect(**PG_CONN)
    df = pd.read_sql("""
        SELECT
            elevation, aspect, slope,
            horizontal_distance_to_hydrology,
            vertical_distance_to_hydrology,
            horizontal_distance_to_roadways,
            hillshade_9am, hillshade_noon, hillshade_3pm,
            horizontal_distance_to_fire_points,
            wilderness_area_1, wilderness_area_2, wilderness_area_3, wilderness_area_4,
            soil_type_1,  soil_type_2,  soil_type_3,  soil_type_4,  soil_type_5,
            soil_type_6,  soil_type_7,  soil_type_8,  soil_type_9,  soil_type_10,
            soil_type_11, soil_type_12, soil_type_13, soil_type_14, soil_type_15,
            soil_type_16, soil_type_17, soil_type_18, soil_type_19, soil_type_20,
            soil_type_21, soil_type_22, soil_type_23, soil_type_24, soil_type_25,
            soil_type_26, soil_type_27, soil_type_28, soil_type_29, soil_type_30,
            soil_type_31, soil_type_32, soil_type_33, soil_type_34, soil_type_35,
            soil_type_36, soil_type_37, soil_type_38, soil_type_39, soil_type_40,
            cover_type
        FROM training_data
    """, conn)
    conn.close()

    print(f"Datos cargados: {len(df)} filas, {len(df.columns)} columnas")

    # Separar features (X) de la variable objetivo (y)
    X = df.drop(columns=["cover_type"])
    y = df["cover_type"]

    # División 80% entrenamiento / 20% evaluación
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Entrenamiento: {len(X_train)} filas | Evaluación: {len(X_test)} filas")

    # Entrenar el modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluar
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy en test: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Guardar el modelo en archivo temporal usando el run_id como nombre único
    run_id = context["run_id"].replace(":", "_").replace("+", "_").replace(" ", "_")
    model_path = f"/tmp/model_{run_id}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Modelo guardado temporalmente en: {model_path}")

    # Métricas que se guardarán junto al modelo en MinIO
    metrics = {
        "accuracy": round(float(accuracy), 4),
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test),
        "n_features": len(X.columns),
        "n_estimators": 100,
        "classes": sorted(y.unique().tolist()),
        "trained_at": datetime.utcnow().isoformat()
    }

    # Pasar ruta y métricas a la siguiente tarea
    context["ti"].xcom_push(key="model_path", value=model_path)
    context["ti"].xcom_push(key="metrics", value=metrics)


def save_to_minio(**context):
    """
    Tarea 3: sube el modelo y sus métricas a MinIO.
    El nombre del archivo incluye el timestamp para versionado.
    Ejemplo: model_20260314_153000.pkl
             model_20260314_153000.json
    """
    model_path = context["ti"].xcom_pull(key="model_path", task_ids="train_model")
    metrics    = context["ti"].xcom_pull(key="metrics",     task_ids="train_model")

    # Nombre versionado por timestamp
    timestamp        = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_filename   = f"model_{timestamp}.pkl"
    metrics_filename = f"model_{timestamp}.json"

    s3 = boto3.client("s3", **MINIO_CONFIG)

    # Crear el bucket si no existe
    try:
        s3.head_bucket(Bucket=BUCKET_NAME)
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            s3.create_bucket(Bucket=BUCKET_NAME)
            print(f"Bucket '{BUCKET_NAME}' creado en MinIO")

    # Subir el modelo
    s3.upload_file(model_path, BUCKET_NAME, model_filename)
    print(f"Modelo subido a MinIO: {BUCKET_NAME}/{model_filename}")

    # Subir las métricas como JSON
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=metrics_filename,
        Body=json.dumps(metrics, indent=2).encode("utf-8"),
        ContentType="application/json"
    )
    print(f"Métricas subidas a MinIO: {BUCKET_NAME}/{metrics_filename}")
    print(f"Accuracy del modelo guardado: {metrics['accuracy']*100:.2f}%")

    # Eliminar el archivo temporal
    os.remove(model_path)
    print("Archivo temporal eliminado.")


# ── Definición del DAG ─────────────────────────────────────────────────────
with DAG(
    dag_id="model_training",
    default_args={
        "owner": "mlops",
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
    },
    description="Entrena un RandomForest con los datos de training_data y guarda el modelo en MinIO",
    schedule_interval=None,   # Se dispara manualmente desde la UI de Airflow
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["training", "model"],
) as dag:

    t1 = PythonOperator(task_id="check_data",    python_callable=check_data)
    t2 = PythonOperator(task_id="train_model",   python_callable=train_model)
    t3 = PythonOperator(task_id="save_to_minio", python_callable=save_to_minio)

    t1 >> t2 >> t3
