# ================================================================
# 4 EJERCICIOS DE PROBABILIDAD CONJUNTA
# ================================================================

# ================================================================
# -SE SELECCIONAN AL AZAR DOS ESTUDIANTES EN UN SALON...
# (3 DE SISTEMAS 2 DE ELECTRONICA 3 DE INDUSTRIAL)
# ================================================================

from math import comb
from fractions import Fraction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os

# --- Datos ---
S = 3
E = 2
I = 3
N = S + E + I
n_select = 2
total = comb(N, n_select)  # total combinaciones

# --- Contar las combinaciones para cada (x,y) ---
counts = {}
for x in range(0, n_select+1):       # x = 0,1,2
    for y in range(0, n_select+1):   # y = 0,1,2
        z = n_select - x - y
        if z < 0:
            continue
        # asegúrate que no pedimos más alumnos de los que hay en cada grupo
        if x <= S and y <= E and z <= I:
            counts[(x, y)] = comb(S, x) * comb(E, y) * comb(I, z)
        else:
            counts[(x, y)] = 0

# Filtrar solo pares con conteo > 0
counts = {k: v for k, v in counts.items() if v > 0}

# --- PMF conjunta como Fraction y decimal ---
pmf_frac = {k: Fraction(v, total) for k, v in counts.items()}
pmf_dec = {k: float(pmf_frac[k]) for k in pmf_frac}

# Mostrar tabla conjunta (fracciones y decimales)
x_vals = [0, 1, 2]
y_vals = [0, 1, 2]

# DataFrame fracciones
df_frac = pd.DataFrame(
    [[str(pmf_frac.get((x, y), Fraction(0, 1))) for y in y_vals] for x in x_vals],
    index=[f'X={x}' for x in x_vals],
    columns=[f'Y={y}' for y in y_vals]
)
# DataFrame decimales
df_dec = pd.DataFrame(
    [[pmf_dec.get((x, y), 0.0) for y in y_vals] for x in x_vals],
    index=[f'X={x}' for x in x_vals],
    columns=[f'Y={y}' for y in y_vals]
)

print("=== PMF conjunta (fracciones) ===")
print(df_frac)
print("\n=== PMF conjunta (decimales) ===")
print(df_dec)

# Exportar PMF conjunta a CSV (decimales)
csv_file = "pmf_conjunta_decimales.csv"
df_dec.to_csv(csv_file)
print(f"\nTabla PMF conjunta (decimales) guardada en: {csv_file}")

# --- Marginales ---
pX = {x: sum(pmf_dec.get((x, y), 0.0) for y in y_vals) for x in x_vals}
pY = {y: sum(pmf_dec.get((x, y), 0.0) for x in x_vals) for y in y_vals}

print("\n=== Marginal P(X) ===")
for x in x_vals:
    # mostrar como fracción si es posible
    frac = sum(pmf_frac.get((x, y), Fraction(0, 1)) for y in y_vals)
    print(f"P(X={x}) = {frac} = {float(frac):.9f}")

print("\n=== Marginal P(Y) ===")
for y in y_vals:
    frac = sum(pmf_frac.get((x, y), Fraction(0, 1)) for x in x_vals)
    print(f"P(Y={y}) = {frac} = {float(frac):.9f}")

# --- Esperanzas, varianzas, covarianza ---
# Esperanzas exactas con Fraction
EX_frac = sum(Fraction(x) * sum(pmf_frac.get((x, y), Fraction(0, 1)) for y in y_vals) for x in x_vals)
EY_frac = sum(Fraction(y) * sum(pmf_frac.get((x, y), Fraction(0, 1)) for x in x_vals) for y in y_vals)

EX = float(EX_frac)
EY = float(EY_frac)

EX2_frac = sum(Fraction(x**2) * sum(pmf_frac.get((x, y), Fraction(0, 1)) for y in y_vals) for x in x_vals)
EY2_frac = sum(Fraction(y**2) * sum(pmf_frac.get((x, y), Fraction(0, 1)) for x in x_vals) for y in y_vals)

VarX_frac = EX2_frac - EX_frac * EX_frac
VarY_frac = EY2_frac - EY_frac * EY_frac

EXY_frac = sum(Fraction(x * y) * pmf_frac.get((x, y), Fraction(0, 1)) for x in x_vals for y in y_vals)
Cov_frac = EXY_frac - EX_frac * EY_frac

# Mostrar resultados exactos y decimales
print("\n=== Resultados exactos (fracciones) ===")
print(f"E[X] = {EX_frac} = {EX:.6f}")
print(f"E[Y] = {EY_frac} = {EY:.6f}")
print(f"E[X^2] = {EX2_frac} = {float(EX2_frac):.6f}")
print(f"E[Y^2] = {EY2_frac} = {float(EY2_frac):.6f}")
print(f"Var(X) = {VarX_frac} = {float(VarX_frac):.6f}")
print(f"Var(Y) = {VarY_frac} = {float(VarY_frac):.6f}")
print(f"E[XY] = {EXY_frac} = {float(EXY_frac):.6f}")
print(f"Cov(X,Y) = {Cov_frac} = {float(Cov_frac):.6f}")

# Correlación
VarX = float(VarX_frac)
VarY = float(VarY_frac)
Cov = float(Cov_frac)
Corr = Cov / (np.sqrt(VarX * VarY)) if VarX > 0 and VarY > 0 else float('nan')
print(f"Corr(X,Y) = {Corr:.6f}")

# --- Gráficas ---
out_dir = "pmf_plots"
os.makedirs(out_dir, exist_ok=True)

# 1) Heatmap PMF conjunta
P = np.array([[pmf_dec.get((x, y), 0.0) for y in y_vals] for x in x_vals])
plt.figure(figsize=(5,5))
plt.imshow(P, origin='lower', extent=[min(y_vals)-0.5, max(y_vals)+0.5, min(x_vals)-0.5, max(x_vals)+0.5])
plt.colorbar()
plt.title("PMF conjunta P(X=x,Y=y)")
plt.xlabel("Y (número de electrónicos)")
plt.ylabel("X (número de sistemas)")
plt.xticks(y_vals)
plt.yticks(x_vals)
fname = os.path.join(out_dir, "heatmap_pmf_conjunta.png")
plt.savefig(fname, bbox_inches='tight')
plt.show()
print(f"Heatmap guardado en: {fname}")

# 2) Barras: marginal P(X)
plt.figure()
plt.bar(x_vals, [pX[x] for x in x_vals])
plt.title("Marginal P(X)")
plt.xlabel("X (número de sistemas)")
plt.ylabel("P(X=x)")
plt.xticks(x_vals)
fname = os.path.join(out_dir, "marginal_PX.png")
plt.savefig(fname, bbox_inches='tight')
plt.show()
print(f"Marginal P(X) guardada en: {fname}")

# 3) Barras: marginal P(Y)
plt.figure()
plt.bar(y_vals, [pY[y] for y in y_vals])
plt.title("Marginal P(Y)")
plt.xlabel("Y (número de electrónicos)")
plt.ylabel("P(Y=y)")
plt.xticks(y_vals)
fname = os.path.join(out_dir, "marginal_PY.png")
plt.savefig(fname, bbox_inches='tight')
plt.show()
print(f"Marginal P(Y) guardada en: {fname}")

# 4) Scatter: puntos (x,y) con tamaño proporcional a la probabilidad conjunta
plt.figure()
xs = []; ys = []; sizes = []
for (x, y), prob in pmf_dec.items():
    xs.append(x); ys.append(y); sizes.append(prob * 2000)
plt.scatter(xs, ys, s=sizes)
for (x, y), prob in pmf_dec.items():
    plt.text(x + 0.05, y + 0.05, f"{prob:.3f}", fontsize=9)
plt.title("Probabilidades conjuntas (tamaño ~ probabilidad)")
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(-0.5, 2.5)
plt.ylim(-0.5, 2.5)
fname = os.path.join(out_dir, "scatter_pmf.png")
plt.savefig(fname, bbox_inches='tight')
plt.show()
print(f"Scatter guardado en: {fname}")

print("\nFin del script. Se generaron las tablas, resultados numéricos y las gráficas (png).")

# ================================================================
# -UNA FABRICA DE DULCES DISTRIBUYE CAJAS...
# VIENE DADA POR LA FUNCION 2/5 (2X+3Y) ; 0<X<1 0<Y<1
# ================================================================
from fractions import Fraction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def f_xy(x, y):
    return (2/5) * (2*x + 3*y)

# Marginales analíticas:
fX = lambda x: (4*x + 3)/5
fY = lambda y: (2 + 6*y)/5

# Resultados exactos (fracciones)
EX = Fraction(17,30)
EY = Fraction(3,5)
EX2 = Fraction(2,5)
EY2 = Fraction(13,30)
VarX = EX2 - EX*EX
VarY = EY2 - EY*EY
EXY = Fraction(1,3)
Cov = EXY - EX*EY

# Imprime resultados
print("E[X] =", EX, float(EX))
print("E[Y] =", EY, float(EY))
print("Var(X) =", VarX, float(VarX))
print("Var(Y) =", VarY, float(VarY))
print("E[XY] =", EXY, float(EXY))
print("Cov(X,Y) =", Cov, float(Cov))
print("Corr =", float(Cov) / np.sqrt(float(VarX)*float(VarY)))

# Probabilidad en B
P_B = Fraction(13,160)
print("P(B) exacta =", P_B, float(P_B))

# Graficas: heatmap y marginales
os.makedirs("plots_ejercicio2", exist_ok=True)
nx = 201
xv = np.linspace(0,1,nx); yv = np.linspace(0,1,nx)
Xv, Yv = np.meshgrid(xv, yv, indexing='xy')
Fv = f_xy(Xv, Yv)
plt.figure(figsize=(5,5))
plt.imshow(Fv, origin='lower', extent=[0,1,0,1], aspect='auto')
plt.colorbar(); plt.title("Heatmap f(x,y)")
plt.xlabel("x"); plt.ylabel("y")
plt.savefig("plots_ejercicio2/heatmap_fxy.png", bbox_inches='tight'); plt.show()

plt.figure()
xx = np.linspace(0,1,201)
plt.plot(xx, [fX(x) for x in xx])
plt.title("f_X(x)"); plt.xlabel("x"); plt.ylabel("f_X(x)")
plt.savefig("plots_ejercicio2/marginal_fX.png", bbox_inches='tight'); plt.show()

plt.figure()
yy = np.linspace(0,1,201)
plt.plot(yy, [fY(y) for y in yy])
plt.title("f_Y(y)"); plt.xlabel("y"); plt.ylabel("f_Y(y)")
plt.savefig("plots_ejercicio2/marginal_fY.png", bbox_inches='tight'); plt.show()

# ================================================================
# -DADA LA FUNCION DE PROBABILIDAD DISCRETA F(X,Y)= X+Y=30
# DETERMINE:
# A) P(X+Y=4)
# B) P(X>Y)
# B) P(X>2 , Y<1)
# B) P(X<2 , Y=1)
# ================================================================

from fractions import Fraction
import numpy as np
import matplotlib.pyplot as plt

# construir pmf
pmf = {}
for x in range(0,4):
    for y in range(0,3):
        pmf[(x,y)] = Fraction(x+y, 30)

# imprimir tabla (fracciones y decimales)
print("Tabla PMF (fracciones):")
for x in range(0,4):
    row = [ str(pmf[(x,y)]) for y in range(0,3) ]
    print(f"x={x}: ", row)
print("\nTabla PMF (decimales):")
for x in range(0,4):
    rowd = [ float(pmf[(x,y)]) for y in range(0,3) ]
    print(f"x={x}: ", ["{:.4f}".format(v) for v in rowd])

# función auxiliar para sumar prob event
def prob(condition):
    return sum(p for (x,y),p in pmf.items() if condition(x,y))

# a) P(X+Y=4)
pa = prob(lambda x,y: x+y==4)
# b) P(X>Y)
pb = prob(lambda x,y: x>y)
# c) P(X>=2, Y<=1)
pc = prob(lambda x,y: (x>=2) and (y<=1))
# d) P(X<=2, Y==1)
pd = prob(lambda x,y: (x<=2) and (y==1))

print("\nResultados (fracciones):")
print("a) P(X+Y=4) =", pa)
print("b) P(X>Y)   =", pb)
print("c) P(X>=2, Y<=1) =", pc)
print("d) P(X<=2, Y=1)  =", pd)

print("\nResultados (decimales):")
print("a) {:.6f}".format(float(pa)))
print("b) {:.6f}".format(float(pb)))
print("c) {:.6f}".format(float(pc)))
print("d) {:.6f}".format(float(pd)))

# Mostrar qué pares se suman en cada evento (para verificación)
def list_pairs(condition):
    return [(x,y) for (x,y) in pmf.keys() if condition(x,y)]

print("\nPares usados:")
print("a) x+y=4 ->", list_pairs(lambda x,y: x+y==4))
print("b) x>y   ->", list_pairs(lambda x,y: x>y))
print("c) x>=2,y<=1 ->", list_pairs(lambda x,y: (x>=2) and (y<=1)))
print("d) x<=2,y=1  ->", list_pairs(lambda x,y: (x<=2) and (y==1)))

# --- Graficas (opcional) ---
# heatmap de la pmf
P = np.zeros((4,3))
for x in range(4):
    for y in range(3):
        P[x,y] = float(pmf[(x,y)])
plt.figure(figsize=(5,4))
plt.imshow(P, origin='lower', cmap='viridis', extent=[-0.5,2.5,-0.5,3.5], aspect='auto')
plt.colorbar(label='P(x,y)')
plt.xticks([0,1,2])
plt.yticks([0,1,2,3])
plt.xlabel('y'); plt.ylabel('x')
plt.title('Heatmap PMF P(X=x,Y=y)')
plt.show()

# scatter con tamaño proporcional a prob
xs=[]; ys=[]; sizes=[]
for (x,y),p in pmf.items():
    xs.append(x); ys.append(y); sizes.append(float(p)*2000)
plt.figure()
plt.scatter(xs, ys, s=sizes)
for (x,y),p in pmf.items():
    plt.text(x+0.05, y+0.05, f"{float(p):.3f}", fontsize=9)
plt.xlim(-0.5,3.5); plt.ylim(-0.5,2.5)
plt.xlabel('x'); plt.ylabel('y')
plt.title('PMF (tamaño ~ probabilidad)')
plt.grid(False)
plt.show()

from fractions import Fraction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# -------------------------
# EJERCICIO A (CONTINUO)
# f(x,y) = (x+y)/3, 0<=x<=2, 0<=y<=1
# -------------------------
def ejercicio_A():
    print("=== EJERCICIO A (continuo) ===")
    print("f(x,y)=(x+y)/3, 0<=x<=2, 0<=y<=1")

    # Verificación de normalización (analítica)
    # ∫_0^1 (x+y) dy = x + 1/2
    # ∫_0^2 (x + 1/2) dx = 3
    # 1/3 * 3 = 1
    print("Verificación integral total = 1  -> OK")

    # Marginales
    print("Marginales:")
    print(" f_X(x) = (2x + 1)/6, 0<=x<=2")
    print(" f_Y(y) = (2 + 2y)/3, 0<=y<=1")

    # Expectation and moments (exact via Fraction)
    EX = Fraction(11,9)
    EY = Fraction(5,9)
    EX2 = Fraction(16,9)
    EY2 = Fraction(7,18)
    VarX = EX2 - EX*EX
    VarY = EY2 - EY*EY
    EXY = Fraction(2,3)
    Cov = EXY - EX*EY

    print("E[X] =", EX, float(EX))
    print("E[Y] =", EY, float(EY))
    print("E[X^2] =", EX2, float(EX2))
    print("E[Y^2] =", EY2, float(EY2))
    print("Var(X) =", VarX, float(VarX))
    print("Var(Y) =", VarY, float(VarY))
    print("E[XY] =", EXY, float(EXY))
    print("Cov(X,Y) =", Cov, float(Cov))
    corr = float(Cov) / math.sqrt(float(VarX) * float(VarY))
    print("Corr(X,Y) ≈", corr)

    # Probabilidad en region B = {0.5<x<1.5, 0.2<y<0.8}
    # inner int in y: 0.6*x + 0.3; outer integrate x from 0.5 to 1.5 -> 0.9; times 1/3 => 0.3
    PB = Fraction(3,10)
    print("P(B) =", PB, float(PB))

    # Graficas: heatmap y marginals
    out_dir = "ejercicioA_plots"
    os.makedirs(out_dir, exist_ok=True)
    nx = 201
    x = np.linspace(0,2,nx)
    y = np.linspace(0,1,nx)
    X, Y = np.meshgrid(x, y, indexing='xy')
    F = (X + Y) / 3.0

    plt.figure(figsize=(5,4))
    plt.imshow(F, origin='lower', extent=[0,2,0,1], aspect='auto')
    plt.colorbar(label='f(x,y)')
    plt.title("EJ A: Heatmap f(x,y)")
    plt.xlabel("x"); plt.ylabel("y")
    plt.savefig(os.path.join(out_dir, "heatmap_ejA_fxy.png"), bbox_inches='tight')
    plt.close()

    # marginal f_X
    xx = np.linspace(0,2,201)
    fX = (2*xx + 1)/6.0
    plt.figure()
    plt.plot(xx, fX)
    plt.title("EJ A: f_X(x)")
    plt.xlabel("x"); plt.ylabel("f_X(x)")
    plt.savefig(os.path.join(out_dir, "marginal_ejA_fX.png"), bbox_inches='tight')
    plt.close()

    # marginal f_Y
    yy = np.linspace(0,1,201)
    fY = (2 + 2*yy)/3.0
    plt.figure()
    plt.plot(yy, fY)
    plt.title("EJ A: f_Y(y)")
    plt.xlabel("y"); plt.ylabel("f_Y(y)")
    plt.savefig(os.path.join(out_dir, "marginal_ejA_fY.png"), bbox_inches='tight')
    plt.close()
    print("EJ A: gráficas guardadas en", out_dir)
    print()

# -------------------------
# EJERCICIO B (DISCRETO)
# P(X=x,Y=y) = (2x + y)/75, x=0..4, y=0..2
# -------------------------
def ejercicio_B():
    print("=== EJERCICIO B (discreto) ===")
    print("P(X=x,Y=y) = (2x+y)/75, x=0..4, y=0..2")

    pmf = {}
    for x in range(0,5):
        for y in range(0,3):
            pmf[(x,y)] = Fraction(2*x + y, 75)

    total = sum(pmf.values())
    print("Suma total probabilidades =", total, float(total))  # debe ser 1

    # Mostrar tabla (decimales)
    df = []
    for x in range(0,5):
        row = [float(pmf[(x,y)]) for y in range(0,3)]
        df.append(row)
    import pandas as pd
    df_dec = pd.DataFrame(df, index=[f'X={x}' for x in range(0,5)], columns=[f'Y={y}' for y in range(0,3)])
    print("Tabla PMF (decimales):")
    print(df_dec)

    # Marginales
    pX = {x: sum(float(pmf[(x,y)]) for y in range(0,3)) for x in range(0,5)}
    pY = {y: sum(float(pmf[(x,y)]) for x in range(0,5)) for y in range(0,3)}
    print("Marginal P_X:", pX)
    print("Marginal P_Y:", pY)

    # Esperanzas exactas
    EX = sum(Fraction(x) * sum(pmf[(x,y)] for y in range(0,3)) for x in range(0,5))
    EY = sum(Fraction(y) * sum(pmf[(x,y)] for x in range(0,5)) for y in range(0,3))
    EX2 = sum(Fraction(x**2) * sum(pmf[(x,y)] for y in range(0,3)) for x in range(0,5))
    EY2 = sum(Fraction(y**2) * sum(pmf[(x,y)] for x in range(0,5)) for y in range(0,3))
    VarX = EX2 - EX*EX
    VarY = EY2 - EY*EY
    EXY = sum(Fraction(x*y) * pmf[(x,y)] for x in range(0,5) for y in range(0,3))
    Cov = EXY - EX*EY

    print("E[X] =", EX, float(EX))
    print("E[Y] =", EY, float(EY))
    print("Var(X) =", VarX, float(VarX))
    print("Var(Y) =", VarY, float(VarY))
    print("E[XY] =", EXY, float(EXY))
    print("Cov =", Cov, float(Cov))
    corr = float(Cov) / math.sqrt(float(VarX) * float(VarY))
    print("Corr ≈", corr)

    # Probabilidades pedidas (ejemplo tipo imagen 3)
    pa = sum(pmf[(x,y)] for x in range(0,5) for y in range(0,3) if x+y==3)
    pb = sum(pmf[(x,y)] for x in range(0,5) for y in range(0,3) if x>y)
    pc = sum(pmf[(x,y)] for x in range(3,5) for y in range(0,2))
    pd = sum(pmf[(x,y)] for x in range(0,3) for y in [2])
    print("a) P(X+Y=3) =", pa, float(pa))
    print("b) P(X>Y)   =", pb, float(pb))
    print("c) P(X>=3, Y<=1) =", pc, float(pc))
    print("d) P(X<=2, Y=2)  =", pd, float(pd))

    # Graficas
    out_dir = "ejercicioB_plots"
    os.makedirs(out_dir, exist_ok=True)
    P = np.zeros((5,3))
    for x in range(5):
        for y in range(3):
            P[x,y] = float(pmf[(x,y)])
    plt.figure(figsize=(5,4))
    plt.imshow(P, origin='lower', aspect='auto', extent=[-0.5,2.5,-0.5,4.5])
    plt.colorbar(label='P(x,y)')
    plt.xlabel('y'); plt.ylabel('x'); plt.title('EJ B: Heatmap PMF')
    plt.savefig(os.path.join(out_dir, "heatmap_ejB.png"), bbox_inches='tight')
    plt.close()

    plt.figure()
    xs=[]; ys=[]; sizes=[]
    for (x,y),p in pmf.items():
        xs.append(x); ys.append(y); sizes.append(float(p)*1500)
    plt.scatter(xs, ys, s=sizes)
    for (x,y),p in pmf.items():
        plt.text(x+0.05, y+0.05, f"{float(p):.3f}", fontsize=9)
    plt.xlabel('x'); plt.ylabel('y'); plt.title('EJ B: pmf (tamaño~prob)')
    plt.savefig(os.path.join(out_dir, "scatter_ejB.png"), bbox_inches='tight')
    plt.close()
    print("EJ B: gráficas guardadas en", out_dir)

if __name__ == "__main__":
    ejercicio_A()
    ejercicio_B()
