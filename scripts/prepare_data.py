"""
Script para descargar el dataset Forest Cover Type desde UCI
y convertirlo al formato "raw" que espera la API del profesor.

El dataset original tiene columnas binarias (One-Hot Encoding).
Este script las convierte a texto (Rawah, Neota, etc.)

Ejecutar UNA sola vez antes de levantar docker-compose.
"""

import urllib.request
import gzip
import csv
import os

# ── Rutas ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, '..', 'data_api', 'data', 'covertype.csv')
DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'

# ── Mapeos de One-Hot → texto ───────────────────────────────────────────────
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

def one_hot_to_label(row, start_col, labels):
    """Convierte columnas One-Hot a texto."""
    for i, label in enumerate(labels):
        if row[start_col + i] == '1':
            return label
    return 'Unknown'

def download_and_convert():
    print("Descargando dataset desde UCI (puede tardar 1-2 minutos)...")
    raw_path = OUTPUT_PATH.replace('covertype.csv', 'covtype.data.gz')

    urllib.request.urlretrieve(DATASET_URL, raw_path)
    print("Descarga completa. Convirtiendo formato...")

    header = [
        'Elevation', 'Aspect', 'Slope',
        'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways',
        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points',
        'Wilderness_Area', 'Soil_Type', 'Cover_Type'
    ]

    rows_written = 0
    with gzip.open(raw_path, 'rt') as infile, open(OUTPUT_PATH, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        writer.writerow(header)

        for row in reader:
            if len(row) < 55:
                continue

            # Columnas 0-9: datos numéricos (se mantienen igual)
            numeric = row[0:10]

            # Columnas 10-13: Wilderness Area (4 columnas binarias)
            wilderness = one_hot_to_label(row, 10, WILDERNESS_AREAS)

            # Columnas 14-53: Soil Type (40 columnas binarias)
            soil = one_hot_to_label(row, 14, SOIL_TYPES)

            # Columna 54: Cover Type
            cover_type = row[54]

            writer.writerow(numeric + [wilderness, soil, cover_type])
            rows_written += 1

    # Limpiar archivo temporal
    os.remove(raw_path)

    print(f"Listo. {rows_written:,} filas guardadas en:")
    print(f"  {os.path.abspath(OUTPUT_PATH)}")

if __name__ == '__main__':
    download_and_convert()
