# Samuel Martinez NC 0096
# CASO 1: IDENTIFICACIÓN DE PROBLEMAS
# DATASET: TITANIC
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Asegurar que la carpeta de resultados exista
os.makedirs("caso1-titanic/resultados", exist_ok=True)

print("🚀 Iniciando análisis del Titanic...\n")

# 1. Cargar el archivo CSV
df = pd.read_csv("caso1-titanic/datos/titanic.csv")
print("===== 1. VISTA PREVIA DE DATOS =====")
print(df.head(3), "\n")

# 2. Conocer las dimensiones
print("===== 2. DIMENSIONES =====")
print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}\n")

# 3. Revisar los tipos de datos
print("===== 3. TIPOS DE DATOS =====")
print(df.dtypes, "\n")

# 4. Identificar y contar datos nulos
print("===== 4. DATOS NULOS (Cantidad) =====")
nulos = df.isnull().sum()
print(nulos[nulos > 0], "\n") # Solo muestra columnas con nulos

# 5. Calcular porcentaje de datos nulos
print("===== 5. PORCENTAJE DE NULOS =====")
porcentaje = (df.isnull().mean() * 100).round(2)
print(porcentaje[porcentaje > 0], "%\n")

# 6. Identificar registros duplicados
duplicados = df.duplicated().sum()
print(f"===== 6. REGISTROS DUPLICADOS: {duplicados} =====\n")

# 7. Detección de Valores Atípicos (Outliers) en la Edad usando IQR
print("===== 7. VALORES ATÍPICOS EN 'AGE' =====")
Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR
atipicos = df[(df['age'] < limite_inferior) | (df['age'] > limite_superior)]
print(f"Se encontraron {len(atipicos)} valores atípicos en la edad.\n")

# 8. Visualización de datos nulos (Heatmap)
plt.figure(figsize=(8, 4))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title("Mapa de calor de datos faltantes (Titanic)")
plt.savefig("caso1-titanic/resultados/nulos_titanic.png")
print("✅ Gráfica de nulos guardada en 'resultados/nulos_titanic.png'")

# 9. Limpieza básica (Copia para no alterar el original)
df_limpio = df.copy()

# a) Eliminar duplicados
df_limpio = df_limpio.drop_duplicates()

# b) Rellenar edad faltante con la mediana (estrategia común)
mediana_edad = df_limpio['age'].median()
df_limpio['age'] = df_limpio['age'].fillna(mediana_edad)

# c) Rellenar 'embark_town' con la moda (valor más frecuente)
moda_embarque = df_limpio['embark_town'].mode()[0]
df_limpio['embark_town'] = df_limpio['embark_town'].fillna(moda_embarque)

# 10. Guardar el dataset limpio
df_limpio.to_csv("caso1-titanic/resultados/titanic_limpio.csv", index=False)
print("✅ Dataset limpio guardado en 'resultados/titanic_limpio.csv'")
print("\n🎉 ¡Análisis del Caso 1 completado con éxito!")
print("Samuel Martinez NC 0096")