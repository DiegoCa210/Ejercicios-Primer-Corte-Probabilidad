# ================================================================
#  EJERCICIOS DE FUNCIONES :DISTRIBUCION DE PROBABILIDAD Y FUNCION ACUMULAATIVA DE PROBABILIDAD.
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Caso continuo A: 0 < x < 6 ----------
a1, b1 = 0.0, 6.0
k1 = 1.0 / ((b1**3 - a1**3)/3.0)

def pdf_continuous(x, k, a, b):
    return np.where((x > a) & (x < b), k * x**2, 0.0)

def cdf_continuous(x, k, a, b):
    c = np.where(x <= a, 0.0,
                 np.where(x >= b, 1.0, k * (x**3 - a**3) / 3.0))
    return c

E1 = k1 * (b1**4 - a1**4) / 4.0
E1_2 = k1 * (b1**5 - a1**5) / 5.0
Var1 = E1_2 - E1**2

x_vals1 = np.linspace(a1-1, b1+1, 500)
pdf_vals1 = pdf_continuous(x_vals1, k1, a1, b1)
cdf_vals1 = cdf_continuous(x_vals1, k1, a1, b1)
df_pdf1 = pd.DataFrame({"x": x_vals1, "pdf": pdf_vals1, "cdf": cdf_vals1})

# ---------- Caso continuo B: 1 < x < 2 ----------
a2, b2 = 1.0, 2.0
k2 = 1.0 / ((b2**3 - a2**3)/3.0)

E2 = k2 * (b2**4 - a2**4) / 4.0
E2_2 = k2 * (b2**5 - a2**5) / 5.0
Var2 = E2_2 - E2**2

x_vals2 = np.linspace(a2-1, b2+1, 500)
pdf_vals2 = pdf_continuous(x_vals2, k2, a2, b2)
cdf_vals2 = cdf_continuous(x_vals2, k2, a2, b2)
df_pdf2 = pd.DataFrame({"x": x_vals2, "pdf": pdf_vals2, "cdf": cdf_vals2})

# ---------- Caso discreto: suma de dos dados ----------
outcomes = []
for i in range(1,7):
    for j in range(1,7):
        outcomes.append(i+j)
from collections import Counter
cnt = Counter(outcomes)
pmf = {s: cnt[s]/36.0 for s in sorted(cnt.keys())}
s_values = sorted(pmf.keys())
cdf_vals = []
cum = 0.0
for s in s_values:
    cum += pmf[s]
    cdf_vals.append(cum)

df_dice = pd.DataFrame({
    "s": s_values,
    "P(S=s)": [pmf[s] for s in s_values],
    "P(S<=s)": cdf_vals
})

E_dice = sum(s*pmf[s] for s in s_values)
E2_dice = sum((s**2)*pmf[s] for s in s_values)
Var_dice = E2_dice - E_dice**2

# ---------- Mostrar resultados numéricos ----------
df_results = pd.DataFrame({
    "Caso": ["Continuo (0,6)", "Continuo (1,2)", "Discreto Suma dos dados"],
    "k": [k1, k2, "—"],
    "E[X]": [E1, E2, E_dice],
    "E[X^2]": [E1_2, E2_2, E2_dice],
    "Var(X)": [Var1, Var2, Var_dice]
})

print("\nRESULTADOS CLAVE:\n", df_results)
print("\nPMF y CDF de la suma de dos dados:\n", df_dice)

# ---------- Gráficas ----------
plt.figure(figsize=(8,4))
plt.plot(x_vals1, pdf_vals1)
plt.title("PDF: f(x)=k x^2 en (0,6)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(x_vals1, cdf_vals1)
plt.title("CDF: F(x) para f(x)=k x^2 en (0,6)")
plt.xlabel("x")
plt.ylabel("F(x)")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(x_vals2, pdf_vals2)
plt.title("PDF: f(x)=k x^2 en (1,2)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(x_vals2, cdf_vals2)
plt.title("CDF: F(x) para f(x)=k x^2 en (1,2)")
plt.xlabel("x")
plt.ylabel("F(x)")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
plt.bar(df_dice["s"], df_dice["P(S=s)"])
plt.title("PMF: P(S=s) suma de dos dados")
plt.xlabel("s")
plt.ylabel("P(S=s)")
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(8,4))
plt.step(df_dice["s"], df_dice["P(S<=s)"], where='post')
plt.title("CDF: F(s)=P(S<=s) suma de dos dados")
plt.xlabel("s")
plt.ylabel("F(s)")
plt.grid(True)
plt.show()
