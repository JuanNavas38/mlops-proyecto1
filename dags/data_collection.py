from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import psycopg2

# ── Configuración ──────────────────────────────────────────────────────────
GROUP_NUMBER = 6

# Dentro de Docker, los servicios se comunican por nombre de contenedor
DATA_API_URL = "http://data-api:80"
PG_CONN = {
    "host": "postgres",
    "port": 5432,
    "dbname": "mlops_db",
    "user": "admin",
    "password": "admin123"
}

# Mapeo para One-Hot Encoding (debe coincidir con prepare_data.py)
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

# ── Tareas ─────────────────────────────────────────────────────────────────

def fetch_data(**context):
    """
    Tarea 1: hace UNA petición a la API y guarda la respuesta en XCom.
    XCom es como un buzón interno de Airflow para pasar datos entre tareas.
    """
    response = requests.get(
        f"{DATA_API_URL}/data",
        params={"group_number": GROUP_NUMBER}
    )

    if response.status_code == 400:
        detail = response.json().get('detail', '')
        raise Exception(f"API respondió 400: {detail}")

    response.raise_for_status()
    data = response.json()

    print(f"Batch {data['batch_number']} recibido con {len(data['data'])} filas")

    # Guardar en XCom para que las tareas siguientes puedan leerlo
    context['ti'].xcom_push(key='api_response', value=data)


def save_raw_data(**context):
    """
    Tarea 2: guarda los datos exactamente como llegaron de la API
    en la tabla raw_data de PostgreSQL.
    """
    data = context['ti'].xcom_pull(key='api_response', task_ids='fetch_data')
    rows = data['data']
    batch_number = data['batch_number']

    conn = psycopg2.connect(**PG_CONN)
    cur = conn.cursor()

    inserted = 0
    for row in rows:
        try:
            cur.execute("""
                INSERT INTO raw_data (
                    elevation, aspect, slope,
                    horizontal_distance_to_hydrology, vertical_distance_to_hydrology,
                    horizontal_distance_to_roadways,
                    hillshade_9am, hillshade_noon, hillshade_3pm,
                    horizontal_distance_to_fire_points,
                    wilderness_area, soil_type, cover_type
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                float(row[0]), float(row[1]), float(row[2]),
                float(row[3]), float(row[4]), float(row[5]),
                float(row[6]), float(row[7]), float(row[8]),
                float(row[9]),
                row[10], row[11], int(row[12])
            ))
            inserted += 1
        except Exception as e:
            print(f"Fila omitida en raw_data: {e}")

    conn.commit()
    cur.close()
    conn.close()
    print(f"raw_data: {inserted} filas insertadas (batch {batch_number})")


def process_data(**context):
    """
    Tarea 3: limpia los datos (elimina filas con valores inválidos)
    y los guarda en processed_data.
    """
    data = context['ti'].xcom_pull(key='api_response', task_ids='fetch_data')
    rows = data['data']
    batch_number = data['batch_number']

    conn = psycopg2.connect(**PG_CONN)
    cur = conn.cursor()

    inserted = 0
    for row in rows:
        try:
            # Rechazar filas con campos vacíos
            if any(v == '' or v is None for v in row):
                continue

            numeric = [float(row[i]) for i in range(10)]
            cover_type = int(row[12])

            # Validar rangos del dataset
            if not (1 <= cover_type <= 7):
                continue
            if not all(0 <= numeric[i] <= 255 for i in [6, 7, 8]):  # hillshades
                continue

            cur.execute("""
                INSERT INTO processed_data (
                    elevation, aspect, slope,
                    horizontal_distance_to_hydrology, vertical_distance_to_hydrology,
                    horizontal_distance_to_roadways,
                    hillshade_9am, hillshade_noon, hillshade_3pm,
                    horizontal_distance_to_fire_points,
                    wilderness_area, soil_type, cover_type
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                numeric[0], numeric[1], numeric[2],
                numeric[3], numeric[4], numeric[5],
                numeric[6], numeric[7], numeric[8], numeric[9],
                row[10], row[11], cover_type
            ))
            inserted += 1
        except Exception as e:
            print(f"Fila omitida en processed_data: {e}")

    conn.commit()
    cur.close()
    conn.close()
    print(f"processed_data: {inserted} filas insertadas (batch {batch_number})")


def prepare_training_data(**context):
    """
    Tarea 4: aplica One-Hot Encoding a wilderness_area y soil_type
    y guarda el resultado en training_data, listo para entrenar el modelo.
    """
    data = context['ti'].xcom_pull(key='api_response', task_ids='fetch_data')
    rows = data['data']
    batch_number = data['batch_number']

    conn = psycopg2.connect(**PG_CONN)
    cur = conn.cursor()

    inserted = 0
    for row in rows:
        try:
            numeric = [float(row[i]) for i in range(10)]
            cover_type = int(row[12])
            wilderness_area = row[10]
            soil_type = row[11]

            # One-Hot Encoding: Wilderness Area → 4 columnas binarias
            wa = [0, 0, 0, 0]
            if wilderness_area in WILDERNESS_AREAS:
                wa[WILDERNESS_AREAS.index(wilderness_area)] = 1

            # One-Hot Encoding: Soil Type → 40 columnas binarias
            st = [0] * 40
            if soil_type in SOIL_TYPES:
                st[SOIL_TYPES.index(soil_type)] = 1

            values = numeric + wa + st + [cover_type]  # 10 + 4 + 40 + 1 = 55 valores

            placeholders = ','.join(['%s'] * 55)
            cur.execute(f"""
                INSERT INTO training_data (
                    elevation, aspect, slope,
                    horizontal_distance_to_hydrology, vertical_distance_to_hydrology,
                    horizontal_distance_to_roadways,
                    hillshade_9am, hillshade_noon, hillshade_3pm,
                    horizontal_distance_to_fire_points,
                    wilderness_area_1, wilderness_area_2, wilderness_area_3, wilderness_area_4,
                    soil_type_1, soil_type_2, soil_type_3, soil_type_4, soil_type_5,
                    soil_type_6, soil_type_7, soil_type_8, soil_type_9, soil_type_10,
                    soil_type_11, soil_type_12, soil_type_13, soil_type_14, soil_type_15,
                    soil_type_16, soil_type_17, soil_type_18, soil_type_19, soil_type_20,
                    soil_type_21, soil_type_22, soil_type_23, soil_type_24, soil_type_25,
                    soil_type_26, soil_type_27, soil_type_28, soil_type_29, soil_type_30,
                    soil_type_31, soil_type_32, soil_type_33, soil_type_34, soil_type_35,
                    soil_type_36, soil_type_37, soil_type_38, soil_type_39, soil_type_40,
                    cover_type
                ) VALUES ({placeholders})
            """, values)
            inserted += 1
        except Exception as e:
            print(f"Fila omitida en training_data: {e}")

    conn.commit()
    cur.close()
    conn.close()
    print(f"training_data: {inserted} filas insertadas (batch {batch_number})")


# ── Definición del DAG ─────────────────────────────────────────────────────
with DAG(
    dag_id='data_collection',
    default_args={
        'owner': 'mlops',
        'retries': 1,
        'retry_delay': timedelta(minutes=1),
    },
    description='Recolecta 1 porción de datos por ejecución y la procesa hasta training_data',
    schedule_interval='*/5 * * * *',  # cada 5 minutos
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['data', 'collection'],
) as dag:

    t1 = PythonOperator(task_id='fetch_data',             python_callable=fetch_data)
    t2 = PythonOperator(task_id='save_raw_data',          python_callable=save_raw_data)
    t3 = PythonOperator(task_id='process_data',           python_callable=process_data)
    t4 = PythonOperator(task_id='prepare_training_data',  python_callable=prepare_training_data)

    # Orden de ejecución
    t1 >> t2 >> t3 >> t4
