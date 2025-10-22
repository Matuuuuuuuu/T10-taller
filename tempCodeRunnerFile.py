from scipy.integrate import ode, solve_ivp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("file_t2.csv")
print(df.head())
print(df.shape)



def SimuladorLotkaVolterra(P, z0, file_t, file_out):
    # 1️⃣ Leer los tiempos desde el archivo file_t
    #t_eval = np.loadtxt(file_t, skiprows=1)
    #t_eval = np.sort(t_eval)
    df = pd.read_csv(file_t)
    t_eval = df.iloc[:,0]
    t_eval = np.sort(t_eval)
    # 2️⃣ Desempaquetar parámetros
    a, b, g, d = P  # alfa, beta, gamma, delta

    # 3️⃣ Definir el sistema usando lambda (sin hardcodear parámetros)
    f = lambda t, z: [
        z[0] * (a - b * z[1]),   # dx/dt
        -z[1] * (g - d * z[0])   # dy/dt
    ]
    tspan = [t_eval.min(), t_eval.max()]
    # 4️⃣ Resolver el sistema
    sol = solve_ivp(f, tspan, z0, t_eval=t_eval)

    # 5️⃣ Guardar resultados: columnas (t, x, y)
    datos = np.column_stack((sol.t, sol.y.T))
    np.savetxt(file_out, datos, header="t x y", comments='')

    # 6️⃣ Graficar x(t) e y(t)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(sol.t, sol.y[0], label="Presas (x)")
    plt.plot(sol.t, sol.y[1], label="Depredadores (y)")
    plt.xlabel("Tiempo")
    plt.ylabel("Población")
    plt.title("Evolución temporal")
    plt.legend()
    plt.grid(True)

    # 7️⃣ Graficar diagrama de fases (x vs y)
    plt.subplot(1, 2, 2)
    plt.plot(sol.y[0], sol.y[1])
    plt.xlabel("Presas (x)")
    plt.ylabel("Depredadores (y)")
    plt.title("Diagrama de fases")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

P= [0.1,0.05,0.1,0.1]
z0 = np.array([10,5])
SimuladorLotkaVolterra(P,z0,"file_t2.csv","file_o.txt")




