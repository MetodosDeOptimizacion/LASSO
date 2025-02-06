import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
import statsmodels.api as sm

# Generación de datos simulados
np.random.seed(42)

# Simulando descriptores para 100 casas
Tamaño = np.random.rand(100) * 150  # en metros cuadrados
Habitaciones = np.random.randint(1, 6, 100)
Antigüedad = np.random.randint(1, 30, 100)
Proximidad = np.random.rand(100) * 15  # en kilómetros al centro

# Calculando el precio con una fórmula lineal y ruido aleatorio
Precio = 3000 * Tamaño + 5000 * Habitaciones - 200 * Antigüedad - 1000 * Proximidad + np.random.normal(0, 20000, 100)

# Crear dataframe
X = np.column_stack((Tamaño, Habitaciones, Antigüedad, Proximidad))
y = Precio

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelos de regresión
# 1. OLS (Regresión Lineal)
ols_model = sm.OLS(y_train, sm.add_constant(X_train)).fit()

# 2. LAD (Regresión de Menor Desviación Absoluta)
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

# 3. PCR (Regresión por Componentes Principales)
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train)
pcr_model = LinearRegression()
pcr_model.fit(X_train_pca, y_train)

# 4. PLS (Regresión de Mínimos Cuadrados Parciales)
pls_model = PLSRegression(n_components=3)
pls_model.fit(X_train, y_train)

# Predicciones y cálculos de R²

# OLS
ols_pred = ols_model.predict(sm.add_constant(X_test))
ols_r2 = r2_score(y_test, ols_pred)

# LAD (Lasso)
lasso_pred = lasso_model.predict(X_test)
lasso_r2 = r2_score(y_test, lasso_pred)

# PCR
X_test_pca = pca.transform(X_test)
pcr_pred = pcr_model.predict(X_test_pca)
pcr_r2 = r2_score(y_test, pcr_pred)

# PLS
pls_pred = pls_model.predict(X_test)
pls_pred = pls_pred.flatten()  # Asegúrate de que sea un array unidimensional
pls_r2 = r2_score(y_test, pls_pred)

# Resultados de R²
print("Resultados de R²:")
print(f"OLS - R²: {ols_r2:.4f}")
print(f"LAD (Lasso) - R²: {lasso_r2:.4f}")
print(f"PCR - R²: {pcr_r2:.4f}")
print(f"PLS - R²: {pls_r2:.4f}")

# Cálculo de Q²
def calc_q2(model, X, y):
    # Realizar la validación cruzada utilizando el error cuadrático medio
    cv_errors = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    mse = -cv_errors  # Convertir los valores a positivos (son negativos por default)
    
    # Calcular Q² con la fórmula
    q2 = 1 - (np.sum(mse) / np.sum((y - np.mean(y)) ** 2))
    return q2

# Q²
lasso_q2 = calc_q2(lasso_model, X_test, y_test)
pcr_q2 = calc_q2(pcr_model, X_test_pca, y_test)
pls_q2 = calc_q2(pls_model, X_test, y_test)

# Mostrar resultados de Q²
print("Resultados de Q²:")
print(f"LAD (Lasso) - Q²: {lasso_q2:.4f}")
print(f"PCR - Q²: {pcr_q2:.4f}")
print(f"PLS - Q²: {pls_q2:.4f}")

# Graficar las predicciones vs los valores reales para cada modelo
def plot_model_results(model_name, real, predicted):
    plt.figure(figsize=(10, 5))
    plt.scatter(real, predicted, color='blue')
    plt.plot([min(real), max(real)], [min(real), max(real)], color='red', linestyle='--')
    plt.title(f"{model_name}: Predicción vs Real")
    plt.xlabel("Valores Reales")
    plt.ylabel("Valores Predichos")
    plt.grid(True)
    plt.show()

# OLS
plot_model_results("OLS", y_test, ols_pred)

# LAD
plot_model_results("LAD", y_test, lasso_pred)

# PCR
plot_model_results("PCR", y_test, pcr_pred)

# PLS
plot_model_results("PLS", y_test, pls_pred)
