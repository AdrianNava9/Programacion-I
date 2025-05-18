import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode, linregress

# Leer datos de un Excel
df = pd.read_excel('galaxias_data.xlsx', sheet_name='data2') 
df.dropna()

# Asignar los datos a variables o arreglos
raefcorkpg = df["raefcorkpg"].to_numpy()
muecorg = df["muecorg"].to_numpy()

# Calcular los datos estadísticos
stats = {
    "Media": np.mean(raefcorkpg),
    "Mediana": np.median(raefcorkpg),
    "Moda": mode(raefcorkpg)[0],
    "Desviación estándar": np.std(raefcorkpg),
    "Varianza": np.var(raefcorkpg),
    "Mínimo": np.min(raefcorkpg),
    "Máximo": np.max(raefcorkpg),
    "Rango": np.max(raefcorkpg)-np.min(raefcorkpg),
    "Covarianza": np.cov(raefcorkpg, muecorg)[0, 1]
}

# Regresión lineal
x = np.log10(raefcorkpg)
y = muecorg
regresion = linregress(x, y)
pendiente = regresion.slope
intercepto = regresion.intercept
r2 = regresion.rvalue ** 2  # Coeficiente de determinación


# Guardar estadísticas y resultados del ajuste en un archivo .txt
with open("resultados.txt", "w") as f:
    f.write("Estadísticas del radio efectivo (raefcorkpg)\n")
    for key, value in stats.items():
        f.write(f"{key}: {value}\n")
    
    f.write("\nRegresión lineal \n")
    f.write(f"Ecuación de la recta: y = {pendiente:.4f} * x + {intercepto:.4f}\n")
    f.write(f"Coeficiente de determinación (R²): {r2:.4f}\n")


# Graficar y guardar como .png
plt.figure()
plt.scatter(x, y, color='blue', s=0.3, label='Datos')
plt.plot(x, pendiente * x + intercepto, color='red', label='Ajuste Lineal')
plt.grid(color='black', linestyle='-', linewidth=0.1)
plt.title('Luminosidad vs log(radio efectivo)', fontsize = 18)
plt.xlabel('log(raefcorkpg)', fontsize = 14)
plt.ylabel('muecorg', fontsize = 14)
plt.legend(fontsize = 12)
plt.tight_layout()
plt.savefig("luminosidad_vs_radio.png")
plt.close()