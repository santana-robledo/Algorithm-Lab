# ## Algoritmo Ternary Search

import matplotlib.pyplot as plt
import numpy as np

# Definir el rango inicial de búsqueda [a0, b0]
Rango = [-4, 8]
a0, b0 = Rango[0], Rango[1]  # Extraer los límites inferior y superior

# Parámetros del algoritmo
presicion = 0.0001  # Precisión deseada para la convergencia
ciclo = 0  # Contador de iteraciones


# Función objetivo que queremos minimizar: f(x) = (x-2)² + 0.5*x
def funcion(x):
    return (x - 2) ** 2 + (0.5 * x)


# Encabezado de la tabla de resultados
print(
    f"{'Ciclo':<6}{'a0':<10}{'b0':<10}{'delta':<10}{'1/3 delta':<12}{'2/3 delta':<12}{'a1':<10}{'b1':<10}{'f(a1)':<12}{'f(b1)':<12}{'Estado':<12}")
print("-" * 110)

# Bucle principal del algoritmo de búsqueda por trisección
while b0 - a0 > presicion:  # Continuar mientras el intervalo sea mayor que la precisión
    # Calcular puntos que dividen el intervalo en tres partes iguales
    a1 = a0 + ((b0 - a0) / 3)  # Punto a 1/3 del intervalo
    b1 = a0 + (2 * ((b0 - a0) / 3))  # Punto a 2/3 del intervalo

    # Comparar los valores de la función en los puntos a1 y b1
    if funcion(a1) > funcion(b1):
        # Si f(a1) > f(b1), el mínimo está en [a1, b0]
        a0 = a1  # Descartar el subintervalo izquierdo [a0, a1]
        estado = "Decrece"  # La función está decreciendo hacia la derecha
    else:
        # Si f(a1) <= f(b1), el mínimo está en [a0, b1]
        b0 = b1  # Descartar el subintervalo derecho [b1, b0]
        estado = "Crece"  # La función está creciendo hacia la derecha

    ciclo += 1  # Incrementar el contador de iteraciones

    # Imprimir resultados de la iteración actual
    print(
        f"{ciclo:<6}{a0:<10.4f}{b0:<10.4f}{(b0 - a0):<10.4f}{(b0 - a0) / 3:<12.4f}{2 * (b0 - a0) / 3:<12.4f}{a1:<10.4f}{b1:<10.4f}{funcion(a1):<12.4f}{funcion(b1):<12.4f}{estado}")

# Una vez alcanzada la precisión deseada, calcular el mínimo aproximado
xmin = (a0 + b0) / 2  # Tomar el punto medio del intervalo final como aproximación
ymin = funcion(xmin)  # Evaluar la función en el punto mínimo

# Mostrar el resultado final
print(f"Mínimo en x ≈ {xmin}, f(x) ≈ {funcion(xmin)}")

# Crear visualización de la función y el mínimo encontrado
x = np.linspace(Rango[0], Rango[1], 400)  # Generar 400 puntos en el rango original
y = funcion(x)  # Calcular los valores de la función en esos puntos

# Configurar y mostrar la gráfica
plt.plot(x, y, label="f(x)")  # Graficar la función
plt.scatter(xmin, ymin, color="red", label=f'Mínimo ≈ ({xmin:.2f}, {ymin:.2f})')  # Marcar el mínimo
plt.title("Gráfica de Función y Mínimo Encontrado")  # Título del gráfico
plt.xlabel("x")  # Etiqueta del eje X
plt.ylabel("f(x)")  # Etiqueta del eje Y
plt.legend()  # Mostrar leyenda
plt.grid(True)  # Activar grid
plt.show()  # Mostrar el gráfico

# for i in range(10):
#     a1 = a0 + ((b0 - a0) / 3)
#     b1 = a0 + 2 * ((b0 - a0) / 3)
#
#     if funcion(a1) > funcion(b1):
#         a0=a1
#
#     else:
#         b0=b1