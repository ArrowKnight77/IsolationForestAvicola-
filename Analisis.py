import os
import pandas as pd
import matplotlib.pyplot as plt


ARCHIVO_RESULTADOS = "resultados_anomalias2.csv"
ARCHIVO_NORMALIZADOS = "datos_normalizados.csv"
CARPETA_SALIDA = "resultados_analisis"


def crear_dir(ruta):
    os.makedirs(ruta, exist_ok=True)


def cargar_csv(ruta):
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    df = pd.read_csv(ruta)
    df.columns = df.columns.str.strip().str.lower()
    return df


def guardar(df, ruta):
    df.to_csv(ruta, encoding="utf-8-sig")
    print(f"Guardado: {ruta}")


def graficar_barras(df, salida):
    conteo = df["anomalia"].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    plt.bar(["Normal", "Anomalía"], [conteo.get(0, 0), conteo.get(1, 0)])
    plt.title("Lecturas normales vs anómalas")
    plt.ylabel("Cantidad")
    plt.tight_layout()
    plt.savefig(salida, dpi=150)
    plt.close()


def graficar_scores(df, salida):
    if "anomaly_score" not in df.columns:
        return

    plt.figure(figsize=(7, 4))
    plt.hist(df["anomaly_score"].dropna(), bins=60)
    plt.title("Distribución del anomaly_score")
    plt.xlabel("Anomaly score")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(salida, dpi=150)
    plt.close()


def graficar_por_hora(df, salida):
    if "hora" not in df.columns:
        return

    temp = df.copy()
    temp["hora"] = pd.to_datetime(temp["hora"], errors="coerce")
    temp = temp.dropna(subset=["hora"])
    temp["hora_dia"] = temp["hora"].dt.hour

    resumen = temp[temp["anomalia"] == 1].groupby("hora_dia").size()

    plt.figure(figsize=(7, 4))
    plt.bar(resumen.index, resumen.values)
    plt.title("Anomalías por hora del día")
    plt.xlabel("Hora")
    plt.ylabel("Cantidad")
    plt.tight_layout()
    plt.savefig(salida, dpi=150)
    plt.close()


def graficar_por_modulo(df, salida):
    if "modulo" not in df.columns:
        return

    resumen = (
        df[df["anomalia"] == 1]
        .groupby("modulo")
        .size()
        .sort_values(ascending=False)
        .head(15)
    )

    if resumen.empty:
        return

    plt.figure(figsize=(9, 4))
    plt.bar(resumen.index.astype(str), resumen.values)
    plt.title("Top módulos con más anomalías")
    plt.xlabel("Módulo")
    plt.ylabel("Cantidad")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(salida, dpi=150)
    plt.close()


def graficar_boxplots(df, variables, carpeta):
    for var in variables:
        normales = df[df["anomalia"] == 0][var].dropna()
        anomalas = df[df["anomalia"] == 1][var].dropna()

        if normales.empty or anomalas.empty:
            continue

        plt.figure(figsize=(6, 4))
        plt.boxplot([normales, anomalas], tick_labels=["Normal", "Anomalía"])
        plt.title(f"{var}: normal vs anomalía")
        plt.ylabel(var)
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta, f"boxplot_{var}.png"), dpi=150)
        plt.close()


def graficar_serie_tiempo(df, variables, carpeta):
    if "hora" not in df.columns:
        return

    temp = df.copy()
    temp["hora"] = pd.to_datetime(temp["hora"], errors="coerce")
    temp = temp.dropna(subset=["hora"]).sort_values("hora")

    if len(temp) > 30000:
        paso = max(1, len(temp) // 30000)
        temp = temp.iloc[::paso]

    for var in variables:
        normales = temp[temp["anomalia"] == 0]
        anomalas = temp[temp["anomalia"] == 1]

        plt.figure(figsize=(11, 4))
        plt.plot(normales["hora"], normales[var], linewidth=0.6, label="Normal")
        plt.scatter(anomalas["hora"], anomalas[var], s=10, label="Anomalía")
        plt.title(f"Serie de tiempo: {var}")
        plt.xlabel("Fecha/Hora")
        plt.ylabel(var)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta, f"serie_{var}.png"), dpi=150)
        plt.close()


def main():
    crear_dir(CARPETA_SALIDA)

    carpeta_graficas = os.path.join(CARPETA_SALIDA, "graficas")
    crear_dir(carpeta_graficas)

    df = cargar_csv(ARCHIVO_RESULTADOS)
    df_norm = cargar_csv(ARCHIVO_NORMALIZADOS)

    if "anomalia" not in df.columns:
        raise ValueError("El archivo resultados_anomalias.csv debe contener la columna 'anomalia'.")

    df["anomalia"] = pd.to_numeric(df["anomalia"], errors="coerce").fillna(0).astype(int)

    total = len(df)
    anomalias = int(df["anomalia"].sum())
    normales = total - anomalias
    porcentaje = (anomalias / total) * 100 if total > 0 else 0

    resumen = pd.DataFrame({
        "metrica": ["total", "normales", "anomalias", "porcentaje_anomalias"],
        "valor": [total, normales, anomalias, round(porcentaje, 4)]
    })

    guardar(resumen, os.path.join(CARPETA_SALIDA, "01_resumen_general.csv"))

    variables = [
        v for v in ["temperatura", "humedad", "co", "co2", "nh3", "o2"]
        if v in df.columns
    ]

    if variables:
        estadisticas = df.groupby("anomalia")[variables].agg(
            ["count", "mean", "median", "std", "min", "max"]
        )
        guardar(estadisticas, os.path.join(CARPETA_SALIDA, "02_estadisticas_variables.csv"))

    if "anomaly_score" in df.columns:
        top = df[df["anomalia"] == 1].sort_values("anomaly_score").head(50)
        guardar(top, os.path.join(CARPETA_SALIDA, "03_top_50_anomalias.csv"))

        score_stats = df.groupby("anomalia")["anomaly_score"].agg(
            ["count", "mean", "median", "std", "min", "max"]
        )
        guardar(score_stats, os.path.join(CARPETA_SALIDA, "04_estadisticas_score.csv"))

    if "modulo" in df.columns:
        por_modulo = (
            df[df["anomalia"] == 1]
            .groupby("modulo")
            .size()
            .sort_values(ascending=False)
        )
        guardar(por_modulo.to_frame("cantidad"), os.path.join(CARPETA_SALIDA, "05_anomalias_por_modulo.csv"))

    if "hora" in df.columns:
        temp = df.copy()
        temp["hora"] = pd.to_datetime(temp["hora"], errors="coerce")
        temp = temp.dropna(subset=["hora"])
        temp["hora_dia"] = temp["hora"].dt.hour

        por_hora = temp[temp["anomalia"] == 1].groupby("hora_dia").size()
        guardar(por_hora.to_frame("cantidad"), os.path.join(CARPETA_SALIDA, "06_anomalias_por_hora.csv"))

    if len(df) == len(df_norm):
        df_norm["anomalia"] = df["anomalia"].values

        columnas_norm = [
            c for c in df_norm.select_dtypes(include="number").columns
            if c not in ["anomalia", "anomaly_score"]
        ]

        alejamiento = (
            df_norm[df_norm["anomalia"] == 1][columnas_norm]
            .abs()
            .mean()
            .sort_values(ascending=False)
            .to_frame("promedio_abs_zscore")
        )

        guardar(alejamiento, os.path.join(CARPETA_SALIDA, "07_variables_mas_alejadas.csv"))
    else:
        print("Aviso: los archivos no tienen la misma cantidad de filas. No se cruzaron con datos_normalizados.")

    graficar_barras(df, os.path.join(carpeta_graficas, "01_normales_vs_anomalias.png"))
    graficar_scores(df, os.path.join(carpeta_graficas, "02_histograma_score.png"))
    graficar_por_hora(df, os.path.join(carpeta_graficas, "03_anomalias_por_hora.png"))
    graficar_por_modulo(df, os.path.join(carpeta_graficas, "04_anomalias_por_modulo.png"))
    graficar_boxplots(df, variables, carpeta_graficas)
    graficar_serie_tiempo(df, variables, carpeta_graficas)

    print("\nAnálisis terminado correctamente.")
    print(f"Revisa la carpeta: {CARPETA_SALIDA}")


if __name__ == "__main__":
    main()