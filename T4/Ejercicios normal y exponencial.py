# ================================================================
# 4 EJERCICIOS DE DISTRIBUCIÓN NORMAL Y EXPONENCIAL CON GRÁFICAS
# ================================================================

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

# ================================================================
# FUNCIONES AUXILIARES
# ================================================================
def normal_pdf(x, mu=0, sigma=1):
    """Función de densidad normal"""
    return (1.0/(sigma*math.sqrt(2*math.pi))) * np.exp(-0.5 * ((x-mu)/sigma)**2)

def normal_cdf(x, mu=0, sigma=1):
    """Función de distribución acumulada normal"""
    return 0.5*(1 + math.erf((x-mu)/(sigma*math.sqrt(2))))

def normal_ppf(p, mu=0, sigma=1, tol=1e-9):
    """Cálculo aproximado del percentil (inversa de la CDF)"""
    lo = mu - 10*sigma
    hi = mu + 10*sigma
    while hi - lo > tol:
        mid = 0.5*(lo+hi)
        if normal_cdf(mid, mu, sigma) < p:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo+hi)

def exp_cdf(t, lam):
    return 1 - np.exp(-lam * t)

def exp_sf(t, lam):
    return np.exp(-lam * t)

# ================================================================
# RESULTADOS GUARDADOS
# ================================================================
results = []

# ================================================================
# EJERCICIO 1: Distribución Normal (A)
# X ~ N(250, 12²)
# ================================================================
mu_A, sigma_A = 250, 12

# a) P(X < 230)
p_A_a = normal_cdf(230, mu_A, sigma_A)

# b) P(245 < X < 270)
p_A_b = normal_cdf(270, mu_A, sigma_A) - normal_cdf(245, mu_A, sigma_A)

# c) Esperanza de muestras > 265 si n=50
p_single_gt_265 = 1 - normal_cdf(265, mu_A, sigma_A)
expected_A_c = 50 * p_single_gt_265

results.extend([
    ("Normal A", "P(X < 230)", p_A_a),
    ("Normal A", "P(245 < X < 270)", p_A_b),
    ("Normal A", "Esperadas >265 (n=50)", expected_A_c)
])

# --- Gráfica ---
xA = np.linspace(mu_A-4*sigma_A, mu_A+4*sigma_A, 400)
pdfA = normal_pdf(xA, mu_A, sigma_A)
plt.figure(figsize=(7,4))
plt.plot(xA, pdfA, label="PDF Normal(250, 12²)")
plt.fill_between(xA, pdfA, where=(xA<230), color='skyblue', alpha=0.4, label="P(X<230)")
plt.fill_between(xA, pdfA, where=((xA>245)&(xA<270)), color='lightgreen', alpha=0.4, label="P(245<X<270)")
plt.axvline(265, color='r', linestyle='--', label="X=265")
plt.title("Distribución Normal A")
plt.xlabel("Resistencia (MPa)")
plt.ylabel("Densidad de probabilidad")
plt.legend()
plt.tight_layout()
plt.show()

# ================================================================
# EJERCICIO 2: Distribución Normal (B)
# X ~ N(72, 8²)
# ================================================================
mu_B, sigma_B = 72, 8

# a) P(X ≥ 80)
p_B_a = 1 - normal_cdf(80, mu_B, sigma_B)

# b) Percentil 90
x_B_b = normal_ppf(0.90, mu_B, sigma_B)

# c) P(media > 75) con n=16
n_B = 16
sigma_mean = sigma_B / math.sqrt(n_B)
p_B_c = 1 - normal_cdf(75, mu_B, sigma_mean)

results.extend([
    ("Normal B", "P(X ≥ 80)", p_B_a),
    ("Normal B", "Percentil 90", x_B_b),
    ("Normal B", "P(𝑋̄ > 75) (n=16)", p_B_c)
])

# --- Gráficas ---
xB = np.linspace(mu_B-4*sigma_B, mu_B+4*sigma_B, 400)
pdfB = normal_pdf(xB, mu_B, sigma_B)
plt.figure(figsize=(7,4))
plt.plot(xB, pdfB, label="PDF Normal(72, 8²)")
plt.fill_between(xB, pdfB, where=(xB>=80), color='salmon', alpha=0.4, label="P(X≥80)")
plt.axvline(x_B_b, color='g', linestyle='--', label="Percentil 90")
plt.title("Distribución Normal B")
plt.xlabel("Puntaje")
plt.ylabel("Densidad de probabilidad")
plt.legend()
plt.tight_layout()
plt.show()

# Distribución de medias
xmean = np.linspace(mu_B-4*sigma_mean, mu_B+4*sigma_mean, 300)
pdf_mean = normal_pdf(xmean, mu_B, sigma_mean)
plt.figure(figsize=(7,4))
plt.plot(xmean, pdf_mean, label="Distribución de medias (n=16)")
plt.fill_between(xmean, pdf_mean, where=(xmean>75), color='orange', alpha=0.4, label="P(𝑋̄ > 75)")
plt.title("Distribución de la media muestral (Normal B)")
plt.xlabel("Media muestral")
plt.ylabel("Densidad de probabilidad")
plt.legend()
plt.tight_layout()
plt.show()

# ================================================================
# EJERCICIO 3: Distribución Exponencial (A)
# T ~ Exp(λ = 1/12)
# ================================================================
lam_A = 1/12.0

# a) P(T < 5)
p_EA_a = exp_cdf(5, lam_A)

# b) P(T > 20)
p_EA_b = exp_sf(20, lam_A)

# c) t donde P(T < t) = 0.95
t_EA_c = -math.log(0.05) / lam_A

results.extend([
    ("Exponencial A", "P(T < 5)", p_EA_a),
    ("Exponencial A", "P(T > 20)", p_EA_b),
    ("Exponencial A", "t con CDF=0.95", t_EA_c)
])

# --- Gráfica ---
tA = np.linspace(0, 60, 400)
pdf_expA = lam_A * np.exp(-lam_A * tA)
plt.figure(figsize=(7,4))
plt.plot(tA, pdf_expA, label="PDF Exp(λ=1/12)")
plt.fill_between(tA, pdf_expA, where=(tA<5), color='lightblue', alpha=0.4, label="P(T<5)")
plt.axvline(20, color='r', linestyle='--', label="T=20")
plt.title("Distribución Exponencial A")
plt.xlabel("Tiempo (minutos)")
plt.ylabel("Densidad de probabilidad")
plt.legend()
plt.tight_layout()
plt.show()

# ================================================================
# EJERCICIO 4: Distribución Exponencial (B)
# T ~ Exp(λ = 0.4)
# ================================================================
lam_B = 0.4

# a) P(T < 1)
p_EB_a = exp_cdf(1, lam_B)

# b) T tal que P(T > T) = 0.05
T_EB_b = -math.log(0.05) / lam_B

# c) P(total ≥ 3 | ≥ 2)
p_EB_c = exp_sf(1, lam_B)  # propiedad sin memoria

results.extend([
    ("Exponencial B", "P(T < 1)", p_EB_a),
    ("Exponencial B", "T con P(T>T)=0.05", T_EB_b),
    ("Exponencial B", "P(total≥3 | ≥2)", p_EB_c)
])

# --- Gráfica ---
tB = np.linspace(0, 12, 400)
pdf_expB = lam_B * np.exp(-lam_B * tB)
plt.figure(figsize=(7,4))
plt.plot(tB, pdf_expB, label="PDF Exp(λ=0.4)")
plt.fill_between(tB, pdf_expB, where=(tB<1), color='lightgreen', alpha=0.4, label="P(T<1)")
plt.axvline(T_EB_b, color='purple', linestyle='--', label="T=percentil 95")
plt.title("Distribución Exponencial B")
plt.xlabel("Tiempo (horas)")
plt.ylabel("Densidad de probabilidad")
plt.legend()
plt.tight_layout()
plt.show()

# ================================================================
# MOSTRAR RESULTADOS
# ================================================================
df = pd.DataFrame(results, columns=["Ejercicio", "Cantidad", "Valor"])
df["Valor (aprox)"] = df["Valor"].apply(lambda v: round(v,6))
print("\n=== RESULTADOS NUMÉRICOS ===")
print(df.to_string(index=False))
