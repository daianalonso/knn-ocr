#pragma once
#include "types.h"

/*
Calcula el autovalor (y autovector correspondiente) de módulo máximo

Parámetros:
----------

mat: const Matrix& mat
    Matriz sobre la que queremos calcular el autovalor

num_iter: unsigned (=5000 por defecto)
    Cantidad de iteraciones a correr

eps: double
    Tolerancia a residuo (opcional)

Devuelve:
--------

pair<double, Vector> donde el primer valor es el autovalor
y el segundo el autovector asociado

Nota:
-------
Los valores de num_iters y eps por defecto pueden no ser una buena combinación
en términos de costo computacional.

*/

std::pair<double, Vector>
    power_iteration(const Matrix& mat, unsigned num_iter=5000, double eps=1e-16);


bool
    are_parallel_vectors(Vector &vector1, Vector &vector2, double eps);

/*
Calcula

Parámetros:
----------

mat: const Matrix& mat
    Matriz sobre la que queremos calcular el autovalor

eigenvalsToCalc: unsigned (=1 por defecto)
    Cantidad de autovectores/autovalores a calcular

num_iter: unsigned (=5000 por defecto)
    Cantidad de iteraciones a correr

eps: double
    Tolerancia a residuo (opcional)
Devuelve:
--------

pair<Vector, Matrix> donde:
    - El primero elemento es un vector de los autovalores
    - El segundo elemento es una matriz cuyas columnas son los autovectores
      correspondientes
*/
std::pair<Eigen::VectorXd, Matrix>
    get_first_eigenvalues(const Matrix& mat, unsigned eigenvalsToCalc=1, unsigned num_iter=5000, double eps=1e-16);
