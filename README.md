# Proyecto 1 MLOps — Orquestación, Entrenamiento y Modelos

**Pontificia Universidad Javeriana — Maestría en Inteligencia Artificial**
Curso: Operaciones de Machine Learning
Grupo: Juan N., Maria Camila Cuella, Jonathan

---

## Descripción

Implementación de un entorno MLOps completo usando Docker Compose. El sistema recolecta automáticamente datos del dataset *Forest Cover Type* a través de una API externa, los procesa en tres etapas (raw → procesado → listo para entrenamiento) usando Airflow, y los almacena en PostgreSQL. Los modelos entrenados se guardan en MinIO y se exponen mediante una Inference API construida con FastAPI.

---

## Arquitectura

```
Data API ──► Airflow (DAG cada 5 min) ──► PostgreSQL (3 etapas)
                                                │
                                                ▼
                                    Entrenamiento (pendiente)
                                                │
                                                ▼
                                     MinIO (modelos .pkl)
                                                │
                                                ▼
                                    Inference API - FastAPI
```

### Servicios

| Servicio               | Puerto | Descripción                                      |
|------------------------|--------|--------------------------------------------------|
| Airflow Webserver      | 8080   | Interfaz web para monitorear y ejecutar DAGs     |
| MinIO Console          | 9001   | Interfaz web para gestionar modelos almacenados  |
| MinIO API              | 9000   | API S3-compatible para subir/bajar modelos       |
| Data API               | 8000   | API local que sirve el dataset (réplica del profesor) |
| PostgreSQL (proyecto)  | 5432   | Base de datos con los datos del pipeline         |

---

## Requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y corriendo
- Git

---

## Cómo levantar el sistema

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd Proyecto_MLOPS
```

### 2. Levantar todos los servicios

```bash
docker compose up -d
```

Esto construye las imágenes y levanta los 7 servicios. La primera vez tarda varios minutos porque descarga las imágenes base y Airflow instala sus dependencias Python.

### 3. Verificar que todo está corriendo

```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

Deberías ver todos los servicios en estado `Up` y los servicios críticos como `(healthy)`.

> **Nota:** `mlops_airflow_init` aparece brevemente y luego desaparece — es normal. Es un contenedor de inicialización que corre una sola vez.

### 4. Acceder a las interfaces

| Interfaz       | URL                    | Usuario  | Contraseña |
|----------------|------------------------|----------|------------|
| Airflow        | http://localhost:8080  | admin    | admin123   |
| MinIO Console  | http://localhost:9001  | minioadmin | minioadmin123 |
| Data API docs  | http://localhost:8000/docs | —    | —          |

---

## Pipeline de datos (DAG `data_collection`)

El DAG se ejecuta automáticamente cada 5 minutos. Cada ejecución realiza **una sola petición** a la API y sigue este flujo:

```
fetch_data → save_raw_data → process_data → prepare_training_data
```

| Tarea                  | Qué hace                                                                 |
|------------------------|--------------------------------------------------------------------------|
| `fetch_data`           | Hace 1 petición a la Data API (grupo 6) y guarda la respuesta en XCom   |
| `save_raw_data`        | Inserta los datos exactamente como llegaron en `raw_data`                |
| `process_data`         | Limpia y valida los datos, los guarda en `processed_data`                |
| `prepare_training_data`| Aplica One-Hot Encoding y guarda en `training_data` (55 columnas)        |

### Esquema de la base de datos

**`raw_data`** — datos sin procesar
10 columnas numéricas + `wilderness_area` (texto) + `soil_type` (texto) + `cover_type`

**`processed_data`** — datos limpios y validados
Mismo esquema que `raw_data`, sin filas con nulos ni rangos inválidos

**`training_data`** — listos para entrenar
10 columnas numéricas + 4 columnas `wilderness_area_1..4` (one-hot) + 40 columnas `soil_type_1..40` (one-hot) + `cover_type`

### Verificar los datos recolectados

```bash
docker exec mlops_postgres psql -U admin -d mlops_db -c \
  "SELECT 'raw_data' AS tabla, COUNT(*) FROM raw_data \
   UNION ALL SELECT 'processed_data', COUNT(*) FROM processed_data \
   UNION ALL SELECT 'training_data', COUNT(*) FROM training_data;"
```

---

## Dataset

Se usa el dataset **Forest Cover Type** (variante modificada). Predice el tipo de cobertura forestal a partir de variables cartográficas.

- **Variable objetivo:** `cover_type` — entero del 1 al 7
- **Features numéricas:** elevation, aspect, slope, distancias a agua/carreteras/puntos de fuego, hillshades
- **Features categóricas:** wilderness_area (4 categorías), soil_type (40 categorías)
- **Fuente:** API en `http://10.43.101.94:8080` (máquina del profesor) o réplica local en `http://localhost:8000`

> El archivo `data_api/data/covertype.csv` (38 MB) contiene los datos completos y está incluido en el repositorio para que la réplica local funcione sin depender de la máquina del profesor.

---

## Estado del proyecto

| Componente                        | Estado         |
|-----------------------------------|----------------|
| Docker Compose (infraestructura)  | Completo       |
| Data API (réplica local)          | Completo       |
| DAG recolección de datos          | Completo       |
| Pipeline PostgreSQL (3 etapas)    | Completo       |
| DAG entrenamiento + guardado MinIO| Pendiente      |
| Inference API (FastAPI)           | Pendiente      |

---

## Apagar el sistema

```bash
docker compose down
```

Los datos en PostgreSQL y MinIO se conservan en volúmenes de Docker. Para borrar todo incluyendo los datos:

```bash
docker compose down -v
```
