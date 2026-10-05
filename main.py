import numpy as np

print("Neurona artificial para decidir si una planta necesita riego")
print("Salida 1 = regar | Salida 0 = no regar\n")

# Cada fila contiene: humedad del suelo (%) y temperatura ambiental (°C)
X = np.array([
    [80, 18], [70, 22], [65, 28], [55, 25], [50, 32],
    [40, 30], [35, 25], [30, 32], [20, 35], [10, 38]
], dtype=float)

y = np.array([
    [0], [0], [0], [0], [0],
    [1], [1], [1], [1], [1]
], dtype=float)

print("Entradas originales (humedad %, temperatura °C):\n", X)
print("Respuestas esperadas:\n", y.ravel())

# La humedad se encuentra aproximadamente entre 0 y 100.
# La temperatura se encuentra aproximadamente entre 0 y 50.
#
# Dividimos cada columna por su valor maximo esperado
# para dejar los datos en una escala cercana a 0 y 1.
escala = np.array([100, 50])
X_normalizado = X / escala

print("\nDatos normalizados:")
print(X_normalizado)


def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def entrenar(tasa_aprendizaje, epocas, mostrar_cada=None):
    """Entrena una neurona con dos pesos y un sesgo. Devuelve pesos, sesgo y error final."""
    rng = np.random.default_rng(7)
    pesos = rng.normal(size=(2, 1))
    sesgo = 0.0

    with np.errstate(over="ignore"):
        for epoca in range(epocas):
            z = X_normalizado @ pesos + sesgo
            probabilidades = sigmoide(z)

            # Error cuadratico medio y sus derivadas
            # solo para esta neurona
            error = probabilidades - y
            gradiente_z = 2 * error * probabilidades * (1 - probabilidades) / len(X_normalizado)
            gradiente_pesos = X_normalizado.T @ gradiente_z
            gradiente_sesgo = np.sum(gradiente_z)

            pesos -= tasa_aprendizaje * gradiente_pesos
            sesgo -= tasa_aprendizaje * gradiente_sesgo

            if mostrar_cada and epoca % mostrar_cada == 0:
                print(f"Epoca {epoca:5} | error: {np.mean(error ** 2):.4f}")

        error_final = np.mean((sigmoide(X_normalizado @ pesos + sesgo) - y) ** 2)
    return pesos, sesgo, error_final


def predecir(humedad_temperatura, pesos, sesgo):
    """Normaliza con la MISMA escala del entrenamiento y devuelve probabilidades."""
    datos = np.atleast_2d(np.array(humedad_temperatura, dtype=float)) / escala
    return sigmoide(datos @ pesos + sesgo).ravel()


# ---------------------------------------------------------------
# Configuracion inicial (prueba base)
# ---------------------------------------------------------------
tasa_aprendizaje = 0.5
epocas = 10000

print(f"\n=== Entrenamiento base: tasa={tasa_aprendizaje}, epocas={epocas} ===")
pesos, sesgo, error_final = entrenar(tasa_aprendizaje, epocas, mostrar_cada=2000)

print("\nPeso de la humedad    :", round(float(pesos[0, 0]), 4))
print("Peso de la temperatura:", round(float(pesos[1, 0]), 4))
print("Sesgo                 :", round(float(sesgo), 4))
print("Error final           :", round(float(error_final), 6))

print("\nProbamos lo aprendido:")
print("Si la salida es >= 0.5 = 1 (regar)")
print("Si es < 0.5 = 0 (no regar)")

probabilidades = sigmoide(X_normalizado @ pesos + sesgo)
respuestas = (probabilidades >= 0.5).astype(int)

for caso, (entrada, probabilidad, respuesta, esperada) in enumerate(
    zip(X, probabilidades.ravel(), respuestas.ravel(), y.ravel()), start=1
):
    print(f"Caso {caso:2} | humedad {entrada[0]:3.0f}% temp {entrada[1]:2.0f}°C "
          f"-> probabilidad {probabilidad:.3f} -> respuesta: {respuesta} (esperada {int(esperada)})")

print("\nSigno de los pesos:")
print("- Peso de humedad", "negativo" if pesos[0, 0] < 0 else "positivo",
      ": a mayor humedad", "baja" if pesos[0, 0] < 0 else "sube", "la probabilidad de regar.")
print("- Peso de temperatura", "negativo" if pesos[1, 0] < 0 else "positivo",
      ": a mayor temperatura", "baja" if pesos[1, 0] < 0 else "sube", "la probabilidad de regar.")

# ---------------------------------------------------------------
# Pruebas con condiciones nuevas
# ---------------------------------------------------------------
nuevos = [(75, 30), (45, 34), (25, 22), (50, 25), (30, 40)]

print("\n=== Predicciones con condiciones nuevas (misma escala) ===")
print("Humedad | Temperatura | Probabilidad | Decision")
for humedad, temperatura in nuevos:
    probabilidad = predecir([humedad, temperatura], pesos, sesgo)[0]
    decision = "Regar (1)" if probabilidad >= 0.5 else "No regar (0)"
    print(f"{humedad:6}% | {temperatura:9}°C | {probabilidad:12.4f} | {decision}")

# ---------------------------------------------------------------
# Experimentos con los parametros (cambia solo un parametro a la vez)
# ---------------------------------------------------------------
experimentos = [
    ("Prueba base", 0.5, 10000),
    ("Pocas epocas", 0.5, 100),
    ("Cantidad intermedia", 0.5, 1000),
    ("Mas epocas", 0.5, 20000),
    ("Tasa pequena", 0.01, 10000),
    ("Tasa moderada", 0.1, 10000),
    ("Tasa alta", 1.0, 10000),
    ("Tasa muy alta", 2.0, 10000),
]

print("\n=== Experimentos ===")
print(f"{'Prueba':<20} {'Tasa':>5} {'Epocas':>6} {'Error final':>12} {'Aciertos':>9} {'P(45%,34C)':>11}")
for nombre, tasa, n_epocas in experimentos:
    p, s, err = entrenar(tasa, n_epocas)
    aciertos = int(np.sum((sigmoide(X_normalizado @ p + s) >= 0.5).astype(int) == y.astype(int)))
    prob_45_34 = predecir([45, 34], p, s)[0]
    print(f"{nombre:<20} {tasa:>5} {n_epocas:>6} {err:>12.6f} {aciertos:>6}/10 {prob_45_34:>11.4f}")

# ---------------------------------------------------------------
# Prueba adicional con el umbral (sin volver a entrenar)
# ---------------------------------------------------------------
print("\n=== Umbrales sobre la neurona base (sin reentrenar) ===")
print("Pesos y sesgo se mantienen:", pesos.ravel().round(4), round(float(sesgo), 4))

umbrales = [0.4, 0.5, 0.6]
datos_umbral = [(f"entrenamiento {i + 1}", X[i]) for i in range(len(X))]
datos_umbral += [(f"nuevo {i + 1}", np.array(n, dtype=float)) for i, n in enumerate(nuevos)]

print(f"{'Dato':<17} {'Humedad':>7} {'Temp':>5} {'Prob':>7} " + " ".join(f"u={u}" for u in umbrales))
cambian = []
for nombre, dato in datos_umbral:
    prob = predecir(dato, pesos, sesgo)[0]
    salidas = [int(prob >= u) for u in umbrales]
    marca = "  <- cambia" if len(set(salidas)) > 1 else ""
    if marca:
        cambian.append(nombre)
    print(f"{nombre:<17} {dato[0]:>6.0f}% {dato[1]:>4.0f}C {prob:>7.4f} " + "   ".join(str(s) for s in salidas) + marca)
print("Casos que cambian de decision:", cambian if cambian else "ninguno")
