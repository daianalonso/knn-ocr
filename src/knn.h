#pragma once
#include <vector>
#include "types.h"
#include <string>

using namespace std;

class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);
    void fit(Matrix X, Vector y);
    Vector predict(Matrix X, const string &votation_method);
private:
    unsigned int neighborsQuantity;
    Matrix trainImagesMatrix;
    Vector testLabelsVector;
    unsigned int predict_one_image(Vector a_imagen_to_predict, const string &votation_method);
};
