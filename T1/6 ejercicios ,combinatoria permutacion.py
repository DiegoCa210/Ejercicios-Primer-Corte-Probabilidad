# ================================================================
# 6 EJERCICIOS DE COMBINATORIA,PERMUTACION,CON O SIN REPETICION.
# ================================================================


import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------
# UTILIDADES COMBINATORIAS
# -------------------------
def C(n, k):
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)

def factorial(n):
    return math.factorial(n)

# -------------------------
# PROBLEMA 1: Selección de 3 estudiantes
# Datos:
E = 8   # Electrónica
S = 3   # Sistemas
I = 9   # Industrial
total = E + S + I
n = 3

print("=== PROBLEMA 1 ===")
print(f"Población: E={E}, S={S}, I={I}, total={total}, seleccionados n={n}\n")

# SIN REEMPLAZO (combinatoria)
space_no_rep = C(total, n)

prob_3E_no_rep = C(E, 3) / space_no_rep
prob_3S_no_rep = C(S, 3) / space_no_rep if S >= 3 else 0.0
prob_2E1S_no_rep = (C(E,2) * C(S,1)) / space_no_rep
prob_atleast1S_no_rep = 1 - C(total - S, n) / space_no_rep
prob_1each_no_rep = (C(E,1) * C(S,1) * C(I,1)) / space_no_rep
prob_order_no_rep = (E/total) * (S/(total-1)) * (I/(total-2))

# CON REEMPLAZO (independientes)
pE = E/total; pS = S/total; pI = I/total

prob_3E_rep = pE**3
prob_3S_rep = pS**3
prob_2E1S_rep = C(3,2) * (pE**2) * pS   # any order
prob_atleast1S_rep = 1 - (1-pS)**3
prob_1each_rep = factorial(3) * pE * pS * pI  # 3! * pE*pS*pI
prob_order_rep = pE * pS * pI  # E then S then I in order

print("Sin reemplazo:")
print(f" P(3 E) = {prob_3E_no_rep:.6f}")
print(f" P(3 S) = {prob_3S_no_rep:.6f}")
print(f" P(2E,1S) = {prob_2E1S_no_rep:.6f}")
print(f" P(al menos 1 S) = {prob_atleast1S_no_rep:.6f}")
print(f" P(1 de cada) = {prob_1each_no_rep:.6f}")
print(f" P(orden E-S-I) = {prob_order_no_rep:.6f}\n")

print("Con reemplazo:")
print(f" P(3 E) = {prob_3E_rep:.6f}")
print(f" P(3 S) = {prob_3S_rep:.6f}")
print(f" P(2E,1S) = {prob_2E1S_rep:.6f}")
print(f" P(al menos 1 S) = {prob_atleast1S_rep:.6f}")
print(f" P(1 de cada, cualquier orden) = {prob_1each_rep:.6f}")
print(f" P(orden E-S-I) = {prob_order_rep:.6f}\n")

# Construir distribución de conteos (E_count, S_count, I_count) para n=3
counts = []
probs_nr = {}
probs_r = {}
for e_count in range(0, n+1):
    for s_count in range(0, n+1-e_count):
        i_count = n - e_count - s_count
        ways = C(E, e_count) * C(S, s_count) * C(I, i_count)
        prob_nr = ways / space_no_rep
        multin_coeff = factorial(n) // (factorial(e_count)*factorial(s_count)*factorial(i_count))
        prob_r = multin_coeff * (pE**e_count) * (pS**s_count) * (pI**i_count)
        key = (e_count, s_count, i_count)
        probs_nr[key] = prob_nr
        probs_r[key] = prob_r
        counts.append((key, prob_nr, prob_r))

# Mostrar tabla resumen (Problema 1)
df1 = pd.DataFrame([{"combination": k, "P_no_replacement": v1, "P_replacement": v2}
                    for (k,(k1,(v1,v2))) in zip(probs_nr.items(), enumerate([(k, (probs_nr[k], probs_r[k])) for k in probs_nr.keys()]))])
# (La construcción anterior produce la tabla; usaremos otra forma sencilla para imprimir)
print("Distribución conjunta de conteos (E,S,I) para n=3 (sin y con reemplazo):")
for k in sorted(probs_nr.keys()):
    print(f" {k}: sin_rep={probs_nr[k]:.6f}, con_rep={probs_r[k]:.6f}")
print()

# GRAFICA PROBLEMA 1: barras comparadas por combinaciones
labels = [str(k) for k in probs_nr.keys()]
vals_nr = [probs_nr[k] for k in probs_nr.keys()]
vals_r = [probs_r[k] for k in probs_nr.keys()]

x = np.arange(len(labels))
width = 0.35
plt.figure(figsize=(10,4))
plt.bar(x - width/2, vals_nr, width, label='Sin reemplazo')
plt.bar(x + width/2, vals_r, width, label='Con reemplazo')
plt.xticks(x, labels, rotation=45)
plt.ylabel("Probabilidad")
plt.title("Distribución de conteos (E,S,I) para n=3")
plt.legend()
plt.tight_layout()
plt.show()

# DIAGRAMA DE ÁRBOL (con reemplazo): dibujar niveles y mostrar probabilidades
def draw_tree_three_levels(pE, pS, pI):
    fig, ax = plt.subplots(figsize=(9,4))
    ax.axis('off')
    # niveles y posiciones
    level_y = [0.9, 0.6, 0.35, 0.05]
    x_positions = {'E':0.15, 'S':0.5, 'I':0.85}
    ax.text(0.5, level_y[0], "Start", ha='center', va='center', bbox=dict(boxstyle="round", fc="w"))
    # primer nivel
    ax.text(x_positions['E'], level_y[1], f"E\np={pE:.3f}", ha='center', va='center', bbox=dict(boxstyle="round", fc="lightgray"))
    ax.text(x_positions['S'], level_y[1], f"S\np={pS:.3f}", ha='center', va='center', bbox=dict(boxstyle="round", fc="lightgray"))
    ax.text(x_positions['I'], level_y[1], f"I\np={pI:.3f}", ha='center', va='center', bbox=dict(boxstyle="round", fc="lightgray"))
    # conectar start
    for k,v in x_positions.items():
        ax.annotate("", xy=(v, level_y[1]+0.03), xytext=(0.5, level_y[0]-0.03), arrowprops=dict(arrowstyle="-", lw=1.0))
    # segundo y tercer nivel simplificados: sólo mostrar ejemplos para no saturar el gráfico
    # generar todas secuencias de longitud 3 y colocar en niveles
    sequences = list(itertools.product(['E','S','I'], repeat=3))
    xs = np.linspace(0.05, 0.95, len(sequences))
    for i, seq in enumerate(sequences):
        prob = 1.0
        for ch in seq:
            prob *= (pE if ch=='E' else pS if ch=='S' else pI)
        ax.text(xs[i], level_y[3], f"{''.join(seq)}\n{prob:.3f}", ha='center', va='center', fontsize=6, bbox=dict(boxstyle="round", fc="white"))
        # connector to level 1 (approx)
        # find parent x
        parent_x = x_positions[seq[0]]
        ax.annotate("", xy=(xs[i], level_y[3]+0.03), xytext=(parent_x, level_y[1]-0.03), arrowprops=dict(arrowstyle="-", lw=0.5, alpha=0.5))
    plt.suptitle("Diagrama de árbol (con reemplazo) - secuencias de longitud 3 (ejemplos)")
    plt.tight_layout()
    plt.show()

draw_tree_three_levels(pE, pS, pI)

# -------------------------
# PROBLEMA 2: Ordenamiento de libros
# 4 Ingeniería (distintos), 6 Inglés (distintos), 2 Física (distintos)
n_eng = 4
n_ingl = 6
n_fis = 2
total_books = n_eng + n_ingl + n_fis

# a) Los libros de cada asignatura deben estar todos juntos:
# Tratar cada bloque como un objeto: 3! formas de ordenar los bloques; dentro de cada bloque permutar
ways2_a = factorial(3) * factorial(n_eng) * factorial(n_ingl) * factorial(n_fis)

# b) Solo los libros de Ingeniería deben estar juntos:
# considerar el bloque de ingeniería como 1 objeto + (n_ingl + n_fis) objetos individuales
ways2_b = factorial(1 + n_ingl + n_fis) * factorial(n_eng)

print("=== PROBLEMA 2 ===")
print(f"a) Formas (cada asignatura junta) = {ways2_a:,}")
print(f"b) Formas (solo Ingeniería junta) = {ways2_b:,}\n")

# Graficar (escala log para que se vea)
plt.figure(figsize=(6,4))
plt.bar(["Cada asignatura junta", "Solo Ingeniería junta"], [ways2_a, ways2_b])
plt.yscale('log')
plt.ylabel("Número de ordenamientos (escala log)")
plt.title("Problema 2: ordenamientos de libros")
plt.tight_layout()
plt.show()

# -------------------------
# PROBLEMA 3: Comité (5 ingenieros, 7 abogados) seleccionar 2 ingenieros y 3 abogados
n_engineers = 5
n_lawyers = 7

ways3_a = C(n_engineers, 2) * C(n_lawyers, 3)   # a) cualquier ingeniero y abogado
ways3_b = C(n_engineers, 2) * C(n_lawyers - 1, 2)  # b) abogado determinado (1 fijo)
excluded_engineers = 2
ways3_c = C(n_engineers - excluded_engineers, 2) * C(n_lawyers, 3)  # c) dos ingenieros determinados no pueden pertenecer

print("=== PROBLEMA 3 ===")
print(f"a) Formas (libre) = {ways3_a}")
print(f"b) Formas (abogado determinado incluido) = {ways3_b}")
print(f"c) Formas (2 ingenieros específicos excluidos) = {ways3_c}\n")

# -------------------------
# PROBLEMA 4: Ordenar en fila 5 E, 2 S, 3 I indistinguibles dentro de su carrera
total_students = 5 + 2 + 3
ways4 = factorial(total_students) // (factorial(5) * factorial(2) * factorial(3))
print("=== PROBLEMA 4 ===")
print(f"Formas (multiconjunto) = {ways4}\n")

# Graficar ejemplo de una disposición (colores por carrera)
colors = {'E':'tab:blue', 'S':'tab:orange', 'I':'tab:green'}
arr = ['E']*5 + ['S']*2 + ['I']*3
plt.figure(figsize=(8,1.2))
for i,cat in enumerate(arr):
    plt.gca().add_patch(plt.Rectangle((i,0),1,1, color=colors[cat]))
plt.xlim(0, total_students)
plt.ylim(0,1)
plt.axis('off')
plt.title(f"Ejemplo de una ordenación (una entre {ways4} posibles)")
plt.tight_layout()
plt.show()

# -------------------------
# PROBLEMA 5: Dados
# a) No obtener total 7 u 11 en ninguno de los 2 lanzamientos de un par de dados
# Una tirada de par de dados produce suma con probas:
p_sum7 = 6/36
p_sum11 = 2/36
p_7or11 = p_sum7 + p_sum11
p_not_7or11 = 1 - p_7or11
p5_a = p_not_7or11**2

# b) obtener tres veces el número 6 en 5 lanzamientos de un dado
n5_b = 5
p_single6 = 1/6
p5_b = C(n5_b, 3) * (p_single6**3) * ((1-p_single6)**(n5_b-3))

print("=== PROBLEMA 5 ===")
print(f"a) P(no 7 ni 11 en 2 tiradas del par) = {p5_a:.6f}")
print(f"b) P(3 veces 6 en 5 lanzamientos) = {p5_b:.6f}\n")

# Graficas: suma de dos dados
sums = np.arange(2,13)
counts = [len([(d1,d2) for d1 in range(1,7) for d2 in range(1,7) if d1+d2 == s]) for s in sums]
probs = np.array(counts) / 36.0
plt.figure(figsize=(7,4))
plt.bar(sums, probs)
plt.bar([7,11], [probs[list(sums).index(7)], probs[list(sums).index(11)]], color='red', alpha=0.6)
plt.xlabel("Suma de dos dados")
plt.ylabel("Probabilidad")
plt.title("Distribución de la suma de dos dados (resaltar 7 y 11)")
plt.xticks(sums)
plt.tight_layout()
plt.show()

# Binomial para 5 lanzamientos (p=1/6)
k_vals = np.arange(0,6)
pmf_k = [C(5,k) * (p_single6**k) * ((1-p_single6)**(5-k)) for k in k_vals]
plt.figure(figsize=(6,4))
plt.bar(k_vals, pmf_k)
plt.bar([3], [pmf_k[3]], color='green', alpha=0.6)
plt.xlabel("Número de '6' en 5 lanzamientos")
plt.ylabel("Probabilidad")
plt.title("Binomial (n=5, p=1/6)")
plt.tight_layout()
plt.show()

# -------------------------
# PROBLEMA 6: Máquina produce 12000 memorias diarias, p=0.03 defectuosas.
# Selección aleatoria de 600 memorias. P(12 defectuosas)
n6 = 600
p6 = 0.03
# Binomial exacta (cuidado con overflow; calcular por logs)
def binom_pmf_log(n, k, p):
    # log form
    logC = math.log(math.comb(n, k))
    logpmf = logC + k*math.log(p) + (n-k)*math.log(1-p)
    return math.exp(logpmf)

p6_exact = binom_pmf_log(n6, 12, p6)
# Poisson approximation lambda = n*p
lam_approx = n6 * p6
p6_poisson = math.exp(-lam_approx) * (lam_approx**12) / factorial(12)

print("=== PROBLEMA 6 ===")
print(f"Binomial exacta P(X=12) (n=600, p=0.03) = {p6_exact:.10e}")
print(f"Poisson approx (lambda=18) P(X=12) = {p6_poisson:.10e}\n")

# Graficar la pmf binomial en rango razonable (0..40)
k_plot = np.arange(0, 41)
pmf_binom = [binom_pmf_log(n6, k, p6) for k in k_plot]
plt.figure(figsize=(10,4))
plt.bar(k_plot, pmf_binom)
plt.axvline(12, color='red', linestyle='--', label='k=12')
plt.xlim(0, 40)
plt.xlabel("k (número de defectuosos en 600)")
plt.ylabel("P(X=k)")
plt.title("Distribución Binomial (n=600, p=0.03) - zona mostrada")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# Resumen final en DataFrame y mostrarlo
# -------------------------
summary_items = [
    ("P1 sin_rep 3E", prob_3E_no_rep),
    ("P1 sin_rep 3S", prob_3S_no_rep),
    ("P1 sin_rep 2E1S", prob_2E1S_no_rep),
    ("P1 sin_rep >=1S", prob_atleast1S_no_rep),
    ("P1 sin_rep 1each", prob_1each_no_rep),
    ("P1 sin_rep order ESI", prob_order_no_rep),
    ("P1 rep 3E", prob_3E_rep),
    ("P1 rep 3S", prob_3S_rep),
    ("P1 rep 2E1S", prob_2E1S_rep),
    ("P1 rep >=1S", prob_atleast1S_rep),
    ("P1 rep 1each", prob_1each_rep),
    ("P1 rep order ESI", prob_order_rep),
    ("P2 a) libros", ways2_a),
    ("P2 b) libros", ways2_b),
    ("P3 a) comite", ways3_a),
    ("P3 b) comite", ways3_b),
    ("P3 c) comite", ways3_c),
    ("P4 orden multiset", ways4),
    ("P5 a) dado pares", p5_a),
    ("P5 b) 3 seis en 5", p5_b),
    ("P6 binom P(X=12)", p6_exact),
    ("P6 poisson approx", p6_poisson)
]

df_summary = pd.DataFrame(summary_items, columns=["Cantidad", "Valor"])
pd.set_option('display.float_format', lambda x: f'{x:.8g}')
print("=== RESUMEN ===")
print(df_summary.to_string(index=False))

print("\nScript finalizado. Si quieres que lo guarde en un archivo .py y te lo descargue, dímelo.")
