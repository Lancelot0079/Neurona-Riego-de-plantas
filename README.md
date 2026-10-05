# Neurona artificial para el riego de plantas

Una sola neurona artificial (dos entradas, dos pesos, un sesgo y activación sigmoide) que decide si una planta necesita riego a partir de la **humedad del suelo** (%) y la **temperatura ambiental** (°C). Salida `1` = regar, salida `0` = no regar. Los datos son didácticos y no constituyen una recomendación agronómica.

La implementación sigue el mismo esquema del perceptrón de la compuerta AND (`perceptron-simple`): suma ponderada, sigmoide, error cuadrático medio y descenso del gradiente escrito a mano con NumPy.

## Ejecución

```bash
uv sync
uv run main.py
```

El script imprime el entrenamiento base, las cinco predicciones nuevas, la tabla de experimentos y la prueba de umbrales. La salida completa de una ejecución está en [`salida_consola.txt`](salida_consola.txt). El entrenamiento es determinista (semilla `7`), así que los números son reproducibles.

## Cómo funciona

1. **Normalización**: `escala = [100, 50]` y `X_normalizado = X / escala`. La misma `escala` se aplica a cualquier dato nuevo.
2. **Neurona**: `z = X_normalizado @ pesos + sesgo`, `probabilidad = sigmoide(z)`.
3. **Aprendizaje**: gradiente del error cuadrático medio; `gradiente_pesos = X_normalizado.T @ gradiente_z`.
4. **Decisión**: `respuesta = (probabilidad >= 0.5).astype(int)`.

## Resultado del entrenamiento base (tasa 0.5, 10000 épocas)

| Parámetro | Valor |
|---|---|
| Peso de la humedad | -19.5032 |
| Peso de la temperatura | 2.3101 |
| Sesgo | 7.4022 |
| Error final (MSE) | 0.019645 |

| Caso | Humedad | Temp. | Esperada | Probabilidad | Respuesta |
|---|---|---|---|---|---|
| 1 | 80 % | 18 °C | 0 | 0.001 | 0 |
| 2 | 70 % | 22 °C | 0 | 0.005 | 0 |
| 3 | 65 % | 28 °C | 0 | 0.018 | 0 |
| 4 | 55 % | 25 °C | 0 | 0.103 | 0 |
| 5 | 50 % | 32 °C | 0 | 0.295 | 0 |
| 6 | 40 % | 30 °C | 1 | 0.728 | 1 |
| 7 | 35 % | 25 °C | 1 | 0.850 | 1 |
| 8 | 30 % | 32 °C | 1 | 0.954 | 1 |
| 9 | 20 % | 35 °C | 1 | 0.994 | 1 |
| 10 | 10 % | 38 °C | 1 | 0.999 | 1 |

**Significado de los signos.** El peso de la humedad es **negativo**: cuanto más húmedo está el suelo, menor es `z` y menor la probabilidad de regar. El peso de la temperatura es **positivo**: cuanto más calor, mayor `z` y mayor probabilidad de regar. La humedad pesa mucho más (|-19.5| frente a 2.3) porque en los datos separa las clases con más claridad que la temperatura.

## Predicciones con condiciones nuevas

Normalizadas con la misma `escala` usada en el entrenamiento.

| Humedad | Temperatura | Probabilidad | Decisión |
|---|---|---|---|
| 75 % | 30 °C | 0.0029 | 0 (no regar) |
| 45 % | 34 °C | 0.5490 | 1 (regar) |
| 25 % | 22 °C | 0.9719 | 1 (regar) |
| 50 % | 25 °C | 0.2325 | 0 (no regar) |
| 30 % | 40 °C | 0.9677 | 1 (regar) |

El caso 45 % / 34 °C queda casi sobre la frontera (0.549): la neurona duda.

## Experimentos con los parámetros

Se cambia un solo parámetro por prueba (semilla y pesos iniciales idénticos).

| Prueba | Tasa | Épocas | Error final | Aciertos | P(45 %, 34 °C) | Aprendizaje |
|---|---|---|---|---|---|---|
| Base | 0.5 | 10000 | 0.019645 | 10/10 | 0.5490 | Rápido y estable |
| Pocas épocas | 0.5 | 100 | 0.165623 | 10/10 | 0.5217 | Insuficiente (aciertos por poco margen) |
| Cantidad intermedia | 0.5 | 1000 | 0.067481 | 10/10 | 0.5793 | Aceptable, aún mejorando |
| Más épocas | 0.5 | 20000 | 0.011705 | 10/10 | 0.5263 | Mejora pequeña frente a la base |
| Tasa pequeña | 0.01 | 10000 | 0.130048 | 10/10 | 0.5388 | Lento |
| Tasa moderada | 0.1 | 10000 | 0.049728 | 10/10 | 0.5853 | Estable, algo lento |
| Tasa alta | 1.0 | 10000 | 0.011705 | 10/10 | 0.5263 | Rápido y estable |
| Tasa muy alta | 2.0 | 10000 | 0.006566 | 10/10 | 0.5082 | Muy rápido; estable en estos datos |

## Prueba adicional con el umbral

Sin reentrenar (pesos `[-19.5032, 2.3101]`, sesgo `7.4022`) se comparan umbrales 0.4, 0.5 y 0.6 sobre los 10 casos de entrenamiento y los 5 nuevos. **Solo cambia un caso**: el nuevo 45 % / 34 °C, con probabilidad 0.5490.

| Umbral | Decisión para 45 % / 34 °C |
|---|---|
| 0.4 | 1 (regar) |
| 0.5 | 1 (regar) |
| 0.6 | 0 (no regar) |

Los demás casos están lejos de la frontera (probabilidades < 0.30 o > 0.72), así que no cambian. Modificar el umbral **no cambia los pesos** porque el umbral se aplica *después* de calcular la probabilidad, solo al convertirla en 0 o 1; el entrenamiento ya terminó y los gradientes jamás vieron ese umbral. Cambia la decisión, no lo aprendido: un umbral bajo hace que la neurona riegue con más facilidad (menos riesgo de dejar secar la planta) y uno alto la hace más conservadora.

## Análisis

1. **¿Por qué normalizar?** La humedad llega a 100 y la temperatura a ~50; sin normalizar, la humedad dominaría `z`, la sigmoide se saturaría desde el inicio (gradientes casi nulos) y habría que usar tasas minúsculas. Con ambas entradas cerca de [0, 1] los pesos son comparables y el entrenamiento es estable.
2. **¿Dónde se usó `X_normalizado` y para qué se conservó `X`?** En la suma ponderada (`z = X_normalizado @ pesos + sesgo`) y en el gradiente de los pesos (`X_normalizado.T @ gradiente_z`). `X` se conserva con los valores originales (% y °C) para mostrar e interpretar los resultados.
3. **100 épocas.** El error quedó en 0.166 (frente a 0.0196 de la base): los diez casos se clasifican bien, pero con probabilidades cercanas a 0.5, es decir, con poca confianza. Acertar no equivale a haber aprendido bien.
4. **¿Más épocas siempre ayudan?** No de forma importante. De 100 a 1000 y de 1000 a 10000 la mejora es grande; de 10000 a 20000 el error solo baja de 0.0196 a 0.0117 y las decisiones son las mismas. Hay rendimientos decrecientes.
5. **Tasa pequeña (0.01).** Los pasos son diminutos: tras 10000 épocas el error sigue en 0.130, peor que la base con solo 1000 épocas. Aprende, pero muy lento.
6. **Tasa alta o muy alta (1.0 y 2.0).** En estos datos *no* hubo inestabilidad: 1.0 con 10000 épocas coincide con 0.5 con 20000 (mismo error 0.011705), porque los pasos escalan linealmente, y 2.0 obtuvo el menor error (0.0066). Con error cuadrático y sigmoide los gradientes son pequeños, lo que tolera tasas altas. Esto no significa que sean seguras en general: en otros problemas o con datos sin normalizar tasas así pueden hacer oscilar o divergir.
7. **Signo del peso de la humedad (negativo, -19.5).** Más humedad empuja la salida hacia 0 (no regar); suelo seco empuja hacia 1.
8. **Signo del peso de la temperatura (positivo, 2.31).** Más calor empuja hacia 1 (regar), reflejando mayor pérdida de agua.
9. **¿Por qué un umbral?** La sigmoide entrega una probabilidad continua en (0, 1), pero la acción (regar o no) es binaria. El umbral convierte la confianza en una decisión concreta, y su valor es una elección de política, no un parámetro aprendido.
10. **Limitaciones.** Una sola neurona solo traza una frontera lineal; solo ve humedad y temperatura (ignora tipo de planta, sol, viento, lluvia prevista, hora, tamaño de maceta); se entrenó con apenas 10 ejemplos didácticos sin conjunto de validación (riesgo de sobreajuste); asume que `escala` cubre el rango real; extrapola con falsa seguridad fuera de lo visto; y no es una recomendación agronómica.

## Estructura del proyecto

```
neurona-riego-plantas/
├── main.py              # neurona, entrenamiento, experimentos y umbrales
├── pyproject.toml       # dependencias (numpy)
├── uv.lock
├── salida_consola.txt   # salida completa de `uv run main.py`
└── README.md
```

La carpeta `.venv` no se versiona (está en `.gitignore`).
