#include <algorithm>
#include <chrono>
#include <iostream>
#include "eigen.h"

using namespace std;

/** Calcula el autovector de módulo máximo. **/
pair<double, Vector> power_iteration(const Matrix &X, unsigned num_iter, double eps) {
    Vector autovecPotencial = Vector::Random(X.cols());
    autovecPotencial = autovecPotencial / autovecPotencial.norm();
    Vector autovecAnterior;
    double eigenvalue;

    unsigned int i = 0;

    do {
        autovecAnterior = autovecPotencial;

        autovecPotencial = (X * autovecPotencial);
        autovecPotencial = autovecPotencial / autovecPotencial.norm();

        i++;

    } while (i < num_iter && !are_parallel_vectors(autovecPotencial, autovecAnterior, eps));

    // Despejamos el autovalor.
    Vector autovecTransformed = (X * autovecPotencial);

    eigenvalue = autovecPotencial.transpose() * autovecTransformed;
    eigenvalue = eigenvalue / (autovecPotencial.transpose() * autovecPotencial);

    return make_pair(eigenvalue, autovecPotencial);
}

bool are_parallel_vectors(Vector &vector1, Vector &vector2, double eps) {
    // Si los vectores están normalizados entonces:
    // <vector1, vector2> = ||vector1|| ||vector2|| cos(alpha) = cos(alpha)
    return ((1 - eps) < vector1.transpose() * vector2) && (vector1.transpose() * vector2 <= 1);
}

pair<Vector, Matrix> get_first_eigenvalues(const Matrix &X, unsigned eigenvalsToCalc, unsigned num_iter, double eps) {
    Matrix A(X);
    Vector eigvalues(eigenvalsToCalc);
    Matrix eigvectors(A.rows(), eigenvalsToCalc);
    pair<double, Vector> powerIterationRet;

    // Chequeamos que la cantidad de autovalores pedida no supere la cantidad de filas.
    assert(eigenvalsToCalc <= A.rows() && "No se pueden calcular más autovalores que filas en la matriz");

    // Obtenemos los primeros n resultados.
    unsigned int i = 0;
    while (i < eigenvalsToCalc) {
            powerIterationRet = power_iteration(A, num_iter, eps);
            double eigenvalue = powerIterationRet.first;
            Vector eigenvector = powerIterationRet.second;

            eigvalues[i] = eigenvalue;
            eigvectors.col(i) = eigenvector;

            A = A - (eigenvalue * (eigenvector * eigenvector.transpose()));

            i++;
    }

    return make_pair(eigvalues, eigvectors);
}
