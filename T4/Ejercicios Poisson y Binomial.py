# ================================================================
# 4 EJERCICIOS DE POISSON Y BINOMIAL
# ================================================================
import math
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# FUNCIONES AUXILIARES
# -------------------------
def poisson_pmf(k, lam):
    return math.exp(-lam) * lam**k / math.factorial(k)

def poisson_cdf(k_max, lam):
    return sum(poisson_pmf(k, lam) for k in range(0, k_max+1))

def binomial_pmf(k, n, p):
    return math.comb(n, k) * (p**k) * ((1-p)**(n-k))

def binomial_cdf(k_max, n, p):
    return sum(binomial_pmf(k, n, p) for k in range(0, k_max+1))

# -------------------------
# EJERCICIO POISSON 1
# Un técnico recibe en promedio 3 solicitudes por día (λ = 3).
# a) P(ninguna solicitud) = P(X = 0)
# b) P(más de dos solicitudes) = P(X > 2)
# -------------------------
lam1 = 3

p1_a = poisson_pmf(0, lam1)                   # P(X=0)
p1_le_2 = poisson_cdf(2, lam1)                # P(X <= 2)
p1_b = 1 - p1_le_2                             # P(X > 2)

print("POISSON 1 (lambda=3):")
print(f"  a) P(X=0) = {p1_a:.6f}")
print(f"  b) P(X>2) = {p1_b:.6f}")
print()

# Gráfica Poisson 1 (PMF) con region marcada
k_vals = np.arange(0, 11)  # hasta 10 llamado suficiente
pmf_vals = [poisson_pmf(k, lam1) for k in k_vals]

plt.figure(figsize=(7,4))
plt.bar(k_vals, pmf_vals)
# sombrear las barras relevantes:
plt.bar([0], [poisson_pmf(0, lam1)])                # P(X=0)
plt.bar(k_vals[k_vals>2], [pmf_vals[i] for i in range(len(k_vals)) if k_vals[i]>2], alpha=0.5)
plt.title("Poisson (λ=3) - PMF")
plt.xlabel("Número de solicitudes X")
plt.ylabel("P(X=k)")
plt.xticks(k_vals)
plt.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()

# -------------------------
# EJERCICIO POISSON 2
# En una línea telefónica se reciben en promedio 6 llamadas por hora (λ = 6).
# a) P(exactamente 4 llamadas) = P(X = 4)
# b) P(al menos 5 llamadas) = P(X >= 5)
# -------------------------
lam2 = 6
p2_a = poisson_pmf(4, lam2)
p2_le_4 = poisson_cdf(4, lam2)
p2_b = 1 - p2_le_4

print("POISSON 2 (lambda=6):")
print(f"  a) P(X=4) = {p2_a:.6f}")
print(f"  b) P(X>=5) = {p2_b:.6f}")
print()

# Gráfica Poisson 2 (PMF) con region marcada
k_vals2 = np.arange(0, 16)
pmf_vals2 = [poisson_pmf(k, lam2) for k in k_vals2]

plt.figure(figsize=(8,4))
plt.bar(k_vals2, pmf_vals2)
# sombrear k=4 y k>=5
plt.bar([4], [poisson_pmf(4, lam2)]) 
plt.bar(k_vals2[k_vals2>=5], [pmf_vals2[i] for i in range(len(k_vals2)) if k_vals2[i]>=5], alpha=0.5)
plt.title("Poisson (λ=6) - PMF")
plt.xlabel("Número de llamadas X")
plt.ylabel("P(X=k)")
plt.xticks(k_vals2)
plt.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()

# -------------------------
# EJERCICIO BINOMIAL 1
# Empresa fabrica bombillos, probabilidad defecto p = 0.05.
# Se seleccionan n = 20.
# a) P(X = 2)
# b) P(al menos 1 defectuoso) = 1 - P(X = 0)
# -------------------------
n1, p_def = 20, 0.05
p3_a = binomial_pmf(2, n1, p_def)
p3_b = 1 - binomial_pmf(0, n1, p_def)

print("BINOMIAL 1 (n=20, p=0.05):")
print(f"  a) P(X=2) = {p3_a:.6f}")
print(f"  b) P(X>=1) = {p3_b:.6f}")
print()

# Gráfica Binomial 1
k_vals3 = np.arange(0, n1+1)
pmf_vals3 = [binomial_pmf(k, n1, p_def) for k in k_vals3]

plt.figure(figsize=(8,4))
plt.bar(k_vals3, pmf_vals3)
# sombrear k=2 y k>=1
plt.bar([2], [binomial_pmf(2, n1, p_def)])
plt.bar(k_vals3[k_vals3>=1], [pmf_vals3[i] for i in range(len(k_vals3)) if k_vals3[i]>=1], alpha=0.5)
plt.title(f"Binomial (n={n1}, p={p_def}) - PMF")
plt.xlabel("Número de bombillos defectuosos X")
plt.ylabel("P(X=k)")
plt.xticks(k_vals3)
plt.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()

# -------------------------
# EJERCICIO BINOMIAL 2 (Test)
# Examen con n = 10 preguntas, 4 opciones cada una, se contesta al azar => p = 1/4 = 0.25
# a) P(X = 3)
# b) P(Como máximo 2) = P(X <= 2)
# -------------------------
n2, p_guess = 10, 0.25
p4_a = binomial_pmf(3, n2, p_guess)
p4_b = binomial_cdf(2, n2, p_guess)

print("BINOMIAL 2 (n=10, p=0.25):")
print(f"  a) P(X=3) = {p4_a:.6f}")
print(f"  b) P(X<=2) = {p4_b:.6f}")
print()

# Gráfica Binomial 2
k_vals4 = np.arange(0, n2+1)
pmf_vals4 = [binomial_pmf(k, n2, p_guess) for k in k_vals4]

plt.figure(figsize=(8,4))
plt.bar(k_vals4, pmf_vals4)
# sombrear k=3 y k<=2
plt.bar([3], [binomial_pmf(3, n2, p_guess)])
plt.bar(k_vals4[k_vals4<=2], [pmf_vals4[i] for i in range(len(k_vals4)) if k_vals4[i]<=2], alpha=0.5)
plt.title(f"Binomial (n={n2}, p={p_guess}) - PMF")
plt.xlabel("Número de respuestas correctas X")
plt.ylabel("P(X=k)")
plt.xticks(k_vals4)
plt.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()

# -------------------------
# FIN
# -------------------------
