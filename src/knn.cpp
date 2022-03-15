#include <iostream>
#include <algorithm>
#include "knn.h"
#include<vector>
#include <string>

using namespace std;


KNNClassifier::KNNClassifier(unsigned int n_neighbors) {
    this->neighborsQuantity = n_neighbors;
}

void KNNClassifier::fit(Matrix X, Vector y) {
    this->trainImagesMatrix = X;
    this->testLabelsVector = y;
}

Vector KNNClassifier::predict(Matrix X, const string &votation_method) {
    Vector ret(X.rows());
    Vector distances(trainImagesMatrix.rows()), distancesCopy(trainImagesMatrix.rows());
    Vector neighbors(10);
    Eigen::Index idx; // Índice en neighbors (corresponderá al label con más vecinos).

    for (unsigned i = 0; i < X.rows(); ++i) {
        neighbors.setZero(); // Llena a neighbors de ceros.

        // distances = (||s1-xi||, ..., ||sm-xi||).
        distances = (trainImagesMatrix.rowwise() - X.row(i)).rowwise().squaredNorm();
        distancesCopy = distances; // Se copia distances en distancesCopy.

        // nth-element deja en distancesCopy(neighbors-1) el elemento que corresponde a esa posición
        // en distancesCopy ordenado de menor a mayor:
        nth_element(
                distancesCopy.data(), distancesCopy.data() + neighborsQuantity - 1,
                distancesCopy.data() + distancesCopy.size()
        );
        // v.data() y v.data()+v.size() "equivalen" a v.begin() y v.end() para Eigen pre 3.4.0.

        for (unsigned j = 0, k = 0; k < neighborsQuantity; j++) {
            if (distances(j) <= distancesCopy(neighborsQuantity - 1)) {
                if (votation_method == "distance") {
                    neighbors(testLabelsVector(j)) += double(1 / distances(j));
                } else { // votation_method = "uniform"
                    neighbors(testLabelsVector(j))++;
                }
                k++; // ... y se incrementa la cantidad de vecinos encontrada hasta el momento.
            }
        }
        // Se guarda en ret(i) la posición del máximo elemento en neighbors, que corresponde al label con
        // mayor cantidad de vecinos:
        neighbors.maxCoeff(&idx);
        ret(i) = idx;
    }
    return ret;
}
