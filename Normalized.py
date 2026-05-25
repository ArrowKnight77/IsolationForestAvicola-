import pandas as pd
from sqlalchemy import create_engine
import json
from sklearn.preprocessing import StandardScaler
import joblib

# -------------------------
# 1. Cargar config
# -------------------------
with open('config.json') as f:
    config = json.load(f)

db = config['mysql']

# -------------------------
# 2. Conexión
# -------------------------
engine = create_engine(
    f"mysql+mysqlconnector://{db['user']}:{db['password']}@{db['host']}/{db['database']}"
)

df = pd.read_sql("SELECT * FROM lecturas", engine)

print(f" Filas cargadas desde BD: {len(df)}")
print(f"Columnas originales: {df.columns.tolist()}")

# -------------------------
# 3. Normalizar nombres
# -------------------------
df.columns = df.columns.str.strip().str.lower()
print(f"Columnas tras normalizar: {df.columns.tolist()}")

# -------------------------
# 4. Limpieza
# -------------------------
df['hora'] = pd.to_datetime(df['hora'])
df = df.sort_values('hora').reset_index(drop=True)
df = df.drop_duplicates()
print(f" Tras drop_duplicates: {len(df)} filas")

# o2 está vacía — se elimina del análisis
print(f" o2 dtype: {df['o2'].dtype}, valores no-nulos: {df['o2'].count()}")
df = df.drop(columns=['o2'])

# Filtros 
df = df[(df['temperatura'] >= 15) & (df['temperatura'] <= 45)]  #Se queda solo con las filas donde la temperatura esté entre 15 y 45 grados". de aqui en adelante es error del sensor 
df = df[(df['humedad'] >= 0)      & (df['humedad'] <= 100)]   #rango de 0 a 100 para evitar lecturas erroneas del sensor 
df = df[df['co']  >= 0]     #evita datos negativos que no existen en gases 
df = df[df['co2'] >= 0]
df = df[df['nh3'] >= 0]
print(f" Tras filtros de rangos: {len(df)} filas") #aqui imprimimos los datos que sobrevivieron a lo filtros que son los datos que importan 

if len(df) == 0:
    raise ValueError(" No quedaron filas tras los filtros.")

# -------------------------
# 5. Feature Engineering
# -------------------------
df['temp_ma5'] = df['temperatura'].rolling(5, min_periods=1).mean()   #Trabaja con el promedio de las cinco lecturas anteriores. Eso evita picos y le da al modelo una visión de tendencia en lugar de ruido
df['co2_ma5']  = df['co2'].rolling(5, min_periods=1).mean() #lo mismo para el CO2 min_periods=1 evita que las primeras filas queden como null
df['delta_temp'] = df['temperatura'].diff() #Calcula cuanto cambio la temperatura entre una lectura y la anterior
df['delta_co2']  = df['co2'].diff() # Igual para CO₂
df['hora_dia']   = df['hora'].dt.hour #Extrae solo la hora del día (0–23) del timestamp completo
df['heat_index'] = df['temperatura'] * df['humedad'] #Crea un índice combinado de temperatura × humedad Este producto captura la manera correcta de ambiente para los pollos  

df = df.replace([float('inf'), float('-inf')], pd.NA)# Reemplaza cualquier infinito con un valor nulo. Así el modelo no se rompe si algún calculo se va mas alla

# -------------------------
# 6. Variables finales 
# ahora si hacemos la lista de los datos que realmente importan dropna borra los Na
features = [
    'temperatura', 'humedad', 'co', 'co2', 'nh3',
    'temp_ma5', 'co2_ma5',
    'delta_temp', 'delta_co2',
    'hora_dia', 'heat_index'
]

print(f"\n Antes de dropna: {len(df)} filas")
print(f" NaN por columna:\n{df[features].isna().sum()}")

df = df.dropna(subset=features)
print(f" Tras dropna: {len(df)} filas")

if len(df) == 0:
    raise ValueError(" No quedaron filas tras dropna.")

X = df[features]
print(f"\n Shape final de X: {X.shape}")

# -------------------------
# 7. Normalización
# pulimos los datos para el modelo con standardscaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(X_scaled, columns=features)

# -------------------------
# 8. Guardar
# -------------------------
joblib.dump(scaler, 'scaler.pkl')
df_scaled.to_csv('datos_normalizados.csv', index=False)

print("\n Normalización lista")
print(df_scaled.describe())