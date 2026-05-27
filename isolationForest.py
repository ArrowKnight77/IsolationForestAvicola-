import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine, text
import joblib
import json

# -------------------------
# 1. Cargar config y conexión
# -------------------------
with open('config.json') as f:
    config = json.load(f)

db = config['mysql']

engine = create_engine(
    f"mysql+mysqlconnector://{db['user']}:{db['password']}@{db['host']}/{db['database']}"
)

# -------------------------
# 2. Cargar datos normalizados
# -------------------------
df_scaled = pd.read_csv('datos_normalizados.csv')

print(f" Datos cargados: {df_scaled.shape}")
print(f" Features: {df_scaled.columns.tolist()}")

# -------------------------
# 3. Entrenar Isolation Forest
# -------------------------
model = IsolationForest(
    n_estimators=100,       #Crea 100 arboles para buscar datos aislados 
    contamination=0.005,     #umbral de datos anomalos 1%
    max_samples='auto',     
    random_state=42,  #Fija la semilla aleatoria dara los mismos resultados
    n_jobs=-1  #usamos todos los nucleos para mas rapido procesamiento
)

print("\n Entrenando modelo...")
model.fit(df_scaled)  #aqui el modelo procesa el entrenamiento
print(" Modelo entrenado")

# -------------------------
# 4. Predecir anomalías
# -------------------------
predicciones = model.predict(df_scaled) #El modelo escupe un arreglo de 1 (normal) y -1 (anomalia).
scores       = model.decision_function(df_scaled)# Te da un puntaje. Entre mas negativo sea el número, más "rara" es la anomalía.

df_scaled['anomalia']     = (predicciones == -1).astype(int) #Convierte los -1 y 1 en True y False. Al agregar .astype(int), los convierte en 1 (es anomalía) y 0 (es normal)
df_scaled['anomaly_score'] = scores
# Guarda los scores de anomalía en el dataframe, imprime un resumen con totales y porcentajes
total     = len(df_scaled)
anomalias = df_scaled['anomalia'].sum()
normales  = total - anomalias

print(f"\n Resultados:")
print(f"   Total lecturas : {total:,}")
print(f"   Normales       : {normales:,} ({normales/total*100:.2f}%)")
print(f"   Anomalías      : {anomalias:,} ({anomalias/total*100:.2f}%)")

# -------------------------
# 5. Recuperar datos originales
# vuelve a leer los datos originales desde la bd para aplicarles el mismo proceso de limpieza que al inicio 
df_original = pd.read_sql("SELECT * FROM lecturas", engine)
df_original.columns = df_original.columns.str.strip().str.lower()
df_original['hora'] = pd.to_datetime(df_original['hora'])
df_original = df_original.sort_values('hora').reset_index(drop=True)
df_original = df_original.drop_duplicates()
df_original = df_original.drop(columns=['o2'])

df_original = df_original[(df_original['temperatura'] >= 15) & (df_original['temperatura'] <= 45)]
df_original = df_original[(df_original['humedad'] >= 0)      & (df_original['humedad'] <= 100)]
df_original = df_original[df_original['co']  >= 0]
df_original = df_original[df_original['co2'] >= 0]
df_original = df_original[df_original['nh3'] >= 0]
df_original = df_original.iloc[1:].reset_index(drop=True)
# Hace una copia de los datos originales y le agrega dos columnas nuevas: la etiqueta que decidio el modelo (normal o anomalia) y el score numérico que la respalda. Así tienes en un solo dataframe tanto los valores reales del sensor como el real del modelo
df_resultado = df_original.copy()
df_resultado['anomalia']      = df_scaled['anomalia'].values
df_resultado['anomaly_score'] = df_scaled['anomaly_score'].values

# -------------------------
# 6. Guardar modelo
# -------------------------
joblib.dump(model, 'isolation_forest.pkl')
print("\n Modelo guardado: isolation_forest2.pkl")

# -------------------------
# 7. Guardar CSV
# -------------------------
df_resultado.to_csv('resultados_anomalias2.csv', index=False)
print(" CSV guardado: resultados_anomalias.csv")

print("\n Muestra de anomalías detectadas:")
print(df_resultado[df_resultado['anomalia'] == 1][
    ['hora', 'modulo', 'temperatura', 'humedad', 'co', 'co2', 'nh3', 'anomaly_score']
].head(10).to_string())

# -------------------------
# 8. Insertar en MySQL
# -------------------------
with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS anomalias_detectadas (
            id INT AUTO_INCREMENT PRIMARY KEY,
            hora DATETIME,
            modulo VARCHAR(50),
            temperatura FLOAT,
            humedad FLOAT,
            co FLOAT,
            co2 FLOAT,
            nh3 FLOAT,
            anomalia TINYINT,
            anomaly_score FLOAT
        )
    """))
    conn.commit()
    print("\n Tabla anomalias_detectadas lista en MySQL")

df_anomalias = df_resultado[df_resultado['anomalia'] == 1][[
    'hora', 'modulo', 'temperatura', 'humedad', 'co', 'co2', 'nh3', 'anomalia', 'anomaly_score'
]]

df_anomalias.to_sql(
    'anomalias_detectadas',
    engine,
    if_exists='append',  #agrega las filas al final de la tabla sin borrar lo que ya había.
    index=False,
    chunksize=1000     #manda en lotes de 1000 en caso de una insercion masiva
)

print(f" {len(df_anomalias):,} anomalías insertadas en MySQL")