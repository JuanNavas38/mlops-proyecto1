from fastapi import FastAPI, HTTPException, Query
from typing import List
from pydantic import BaseModel, Field
import random
import json
import time
import csv
import os

# Igual que el entorno real del profesor
MIN_UPDATE_TIME = 300

app = FastAPI(title="Data API - Forest Cover Type", version="1.0.0")

class BatchResponse(BaseModel):
    group_number: int = Field(..., ge=1, le=11)
    batch_number: int
    data: List[List[str]]

# Cargar el dataset al arrancar
data = []
with open('/data/covertype.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # saltar encabezado
    for row in reader:
        data.append(row)

batch_size = len(data) // 10

def get_batch_data(batch_number: int):
    start_index = batch_number * batch_size
    end_index = start_index + batch_size
    return random.sample(data[start_index:end_index], batch_size // 10)

# Cargar timestamps guardados o inicializar
if os.path.isfile('/data/timestamps.json'):
    with open('/data/timestamps.json', "r") as f:
        timestamps = json.load(f)
else:
    timestamps = {str(g): [0, -1] for g in range(1, 11)}

@app.get("/")
async def root():
    return {"Proyecto 2": "Extracción de datos, entrenamiento de modelos."}

@app.get("/data", response_model=BatchResponse)
async def read_data(
    group_number: int = Query(..., ge=1, le=11, description="Número de grupo (1-10)")
):
    global timestamps

    if group_number < 1 or group_number > 11:
        raise HTTPException(status_code=400, detail="Número de grupo inválido")

    if timestamps[str(group_number)][1] >= 11:
        raise HTTPException(status_code=400, detail="Ya se recolectó toda la información mínima necesaria")

    current_time = time.time()
    last_update_time = timestamps[str(group_number)][0]

    if current_time - last_update_time > MIN_UPDATE_TIME:
        timestamps[str(group_number)][0] = current_time
        timestamps[str(group_number)][1] += 2 if timestamps[str(group_number)][1] == -1 else 1

    random_data = get_batch_data(timestamps[str(group_number)][1] % 10)

    with open('/data/timestamps.json', 'w') as file:
        file.write(json.dumps(timestamps))

    return {
        "group_number": group_number,
        "batch_number": timestamps[str(group_number)][1],
        "data": random_data
    }

@app.get("/restart_data_generation")
async def restart_data(
    group_number: int = Query(..., ge=1, le=11)
):
    if group_number < 1 or group_number > 11:
        raise HTTPException(status_code=400, detail="Número de grupo inválido")

    timestamps[str(group_number)][0] = 0
    timestamps[str(group_number)][1] = -1

    with open('/data/timestamps.json', 'w') as file:
        file.write(json.dumps(timestamps))

    return {"ok": True}
