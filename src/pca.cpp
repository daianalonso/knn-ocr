#include <iostream>
#include "pca.h"
#include "eigen.h"

using namespace std;


PCA::PCA(unsigned int n_components) {
    this->n_components = n_components;
}

void PCA::fit(Matrix X) {
    // 1) Calcular la media de cada columna de X.
//    Vector vectorDivisores(X.rows());
//    double divisorMedia = (double)1 / (double)X.rows();
//
//    vectorDivisores.fill(divisorMedia);
//
//    Matrix X_t = X.transpose();
//    Vector vectorMedias = X_t * vectorDivisores;
//
//    // 2) Construir la matriz de covarianza.
//    double divisorCovarianza = (double)1 / (double)(X.rows() - 1);
//    Matrix Y(X);
//    for (unsigned int i = 0; i < Y.rows(); i++)
//        Y.row(i) = Y.row(i) - vectorMedias.transpose();

    X = (X.rowwise() - X.colwise().mean()) / sqrt(X.rows()-1);

    Matrix MatrizCovarianza = X.transpose() * X;
//    MatrizCovarianza = MatrizCovarianza * divisorCovarianza;

    pair<Vector, Matrix> eigens = get_first_eigenvalues(MatrizCovarianza, this->n_components, 5000, 1e-16);
    this->matrizDeCambioDeBase = eigens.second;
}


MatrixXd PCA::transform(Matrix X) {
    return X * this->matrizDeCambioDeBase;
}
