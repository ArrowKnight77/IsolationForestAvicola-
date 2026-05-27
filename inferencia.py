import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import joblib
import json
import time
from datetime import datetime

# -------------------------
# 1. Cargar config y modelos
# -------------------------
with open('config.json') as f:
    config = json.load(f)

db = config['mysql']

engine = create_engine(
    f"mysql+mysqlconnector://{db['user']}:{db['password']}@{db['host']}/{db['database']}"
)

model  = joblib.load('isolation_forest.pkl')
scaler = joblib.load('scaler.pkl')

print(" Modelo y scaler cargados")

# -------------------------
# 2. Configuración
# -------------------------
INTERVALO_SEGUNDOS = 10   # cada cuántos segundos revisa nuevas lecturas
FEATURES = [
    'temperatura', 'humedad', 'co', 'co2', 'nh3',
    'temp_ma5', 'co2_ma5',
    'delta_temp', 'delta_co2',
    'hora_dia', 'heat_index'
]

# -------------------------
# 3. Crear tabla de alertas si no existe
# -------------------------
with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS alertas_tiempo_real (
            id            INT AUTO_INCREMENT PRIMARY KEY,
            id_lectura    INT,
            hora          DATETIME,
            modulo        VARCHAR(50),
            temperatura   FLOAT,
            humedad       FLOAT,
            co            FLOAT,
            co2           FLOAT,
            nh3           FLOAT,
            anomaly_score FLOAT,
            detectado_en  DATETIME
        )
    """))
    conn.commit()
print(" Tabla alertas_tiempo_real lista\n")

# -------------------------
# 4. Obtener el último id procesado
# -------------------------
def get_ultimo_id_procesado():
    """Devuelve el último id_lectura que ya fue evaluado."""
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT MAX(id_lectura) FROM alertas_tiempo_real"
        )).fetchone()
    ultimo = result[0]
    return ultimo if ultimo is not None else 0

# -------------------------
# 5. Leer lecturas nuevas desde la BD
# -------------------------
def get_lecturas_nuevas(ultimo_id, ventana=50):
    """
    Trae lecturas con id > ultimo_id.
    Trae 'ventana' filas extra antes para calcular rolling y diff correctamente.
    """
    query = text("""
        SELECT * FROM lecturas
        WHERE id_lectura > :desde
        ORDER BY id_lectura ASC
        LIMIT 5000
    """)
    # Para rolling/diff necesitamos contexto previo
    query_contexto = text("""
        SELECT * FROM lecturas
        WHERE id_lectura <= :desde
        ORDER BY id_lectura DESC
        LIMIT :ventana
    """)
    with engine.connect() as conn:
        df_nuevas   = pd.read_sql(query, conn, params={'desde': ultimo_id})
        df_contexto = pd.read_sql(query_contexto, conn,
                                  params={'desde': ultimo_id, 'ventana': ventana})

    if df_nuevas.empty:
        return None, None

    # Combinar contexto + nuevas para calcular features correctamente
    df_contexto = df_contexto.iloc[::-1].reset_index(drop=True)  # invertir orden
    df_completo = pd.concat([df_contexto, df_nuevas], ignore_index=True)

    return df_nuevas, df_completo

# -------------------------
# 6. Preparar features (mismo proceso que Normalized.py)
# -------------------------
def preparar_features(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    df['hora'] = pd.to_datetime(df['hora'])
    df = df.sort_values('hora').reset_index(drop=True)

    # Eliminar o2 (vacía en BD)
    if 'o2' in df.columns:
        df = df.drop(columns=['o2'])

    # Filtros
    df = df[(df['temperatura'] >= 15) & (df['temperatura'] <= 45)]
    df = df[(df['humedad'] >= 0)      & (df['humedad'] <= 100)]
    df = df[df['co']  >= 0]
    df = df[df['co2'] >= 0]
    df = df[df['nh3'] >= 0]

    if df.empty:
        return None

    # Features
    df['temp_ma5']  = df['temperatura'].rolling(5, min_periods=1).mean()
    df['co2_ma5']   = df['co2'].rolling(5, min_periods=1).mean()
    df['delta_temp'] = df['temperatura'].diff().fillna(0)
    df['delta_co2']  = df['co2'].diff().fillna(0)
    df['hora_dia']   = df['hora'].dt.hour
    df['heat_index'] = df['temperatura'] * df['humedad']

    df = df.replace([float('inf'), float('-inf')], pd.NA)
    df = df.dropna(subset=FEATURES)

    return df

# -------------------------
# 7. Insertar anomalías en BD
# -------------------------
def insertar_anomalias(df_anomalias):
    if df_anomalias.empty:
        return

    df_insert = df_anomalias[[
        'id_lectura', 'hora', 'modulo',
        'temperatura', 'humedad', 'co', 'co2', 'nh3',
        'anomaly_score'
    ]].copy()
    df_insert['detectado_en'] = datetime.now()

    df_insert.to_sql(
        'alertas_tiempo_real',
        engine,
        if_exists='append',
        index=False,
        chunksize=500
    )

# -------------------------
# 8. Loop principal
# -------------------------
print(f" Iniciando inferencia cada {INTERVALO_SEGUNDOS}s — Ctrl+C para detener\n")

ultimo_id = get_ultimo_id_procesado()
print(f" Último id_lectura procesado: {ultimo_id}")

while True:
    try:
        ahora = datetime.now().strftime('%H:%M:%S')

        # Leer nuevas lecturas
        df_nuevas, df_completo = get_lecturas_nuevas(ultimo_id)

        if df_nuevas is None:
            print(f"[{ahora}] ⏳ Sin lecturas nuevas — esperando...")
        else:
            # Preparar features sobre el bloque completo (contexto + nuevas)
            df_features = preparar_features(df_completo)

            if df_features is None or df_features.empty:
                print(f"[{ahora}] ⚠️  Sin datos válidos tras limpieza")
            else:
                # Quedarse solo con las filas que son nuevas
                ids_nuevos = df_nuevas['id_lectura'].values
                df_solo_nuevas = df_features[
                    df_features['id_lectura'].isin(ids_nuevos)
                ].copy()

                if df_solo_nuevas.empty:
                    print(f"[{ahora}] ⚠️  Nuevas lecturas no pasaron filtros")
                else:
                    # Normalizar
                    X = df_solo_nuevas[FEATURES]
                    X_scaled = scaler.transform(X)

                    # Predecir
                    predicciones = model.predict(X_scaled)
                    scores       = model.decision_function(X_scaled)

                    df_solo_nuevas['anomalia']     = (predicciones == -1).astype(int)
                    df_solo_nuevas['anomaly_score'] = scores

                    # Filtrar anomalías
                    df_anomalias = df_solo_nuevas[df_solo_nuevas['anomalia'] == 1]

                    total     = len(df_solo_nuevas)
                    n_anomal  = len(df_anomalias)

                    print(f"[{ahora}]  {total} lecturas nuevas — "
                          f" {n_anomal} anomalías detectadas")

                    # Mostrar anomalías en consola
                    if n_anomal > 0:
                        print(df_anomalias[[
                            'hora', 'modulo', 'temperatura',
                            'humedad', 'co', 'co2', 'nh3', 'anomaly_score'
                        ]].to_string(index=False))
                        print()

                        # Guardar en BD
                        insertar_anomalias(df_anomalias)

                    # Actualizar último id procesado
                    ultimo_id = int(df_solo_nuevas['id_lectura'].max())

        time.sleep(INTERVALO_SEGUNDOS)

    except KeyboardInterrupt:
        print("\n Inferencia detenida manualmente")
        break
    except Exception as e:
        print(f"[{ahora}]  Error: {e}")
        time.sleep(INTERVALO_SEGUNDOS)
        continue