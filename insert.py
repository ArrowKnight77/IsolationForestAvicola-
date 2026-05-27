import pandas as pd
from sqlalchemy import create_engine
import json

CSV_PATH = "datos_pollitos.csv"  # debe estar en la misma carpeta que este script

# Cargar config.json
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

db = config["mysql"]

engine = create_engine(
    f"mysql+mysqlconnector://{db['user']}:{db['password']}@{db['host']}/{db['database']}"
)

# Columnas esperadas según tu tabla
columnas = [
    "id_lectura",
    "modulo",
    "hora",
    "temperatura",
    "humedad",
    "o2",
    "co",
    "co2",
    "nh3"
]

print("Leyendo CSV...")

df = pd.read_csv(
    CSV_PATH,
    encoding="utf-8",
    na_values=["NULL", "null", "", "NaN"],
)

# Normalizar nombres de columnas
df.columns = df.columns.str.strip().str.lower()

print("Columnas detectadas:", df.columns.tolist())

# Validar columnas
faltantes = [c for c in columnas if c not in df.columns]
if faltantes:
    raise ValueError(f"Faltan columnas en el CSV: {faltantes}")

df = df[columnas]

# Convertir tipos
df["hora"] = pd.to_datetime(df["hora"], errors="coerce")
df["modulo"] = pd.to_numeric(df["modulo"], errors="coerce")

for col in ["temperatura", "humedad", "o2", "co", "co2", "nh3"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Eliminar filas sin datos clave
df = df.dropna(subset=["id_lectura", "modulo", "hora"])

print(f"Filas listas para importar: {len(df):,}")

# Importar por bloques
df.to_sql(
    "lecturas",
    engine,
    if_exists="append",
    index=False,
    chunksize=5000,
    method="multi"
)

print("Importación completada correctamente.")