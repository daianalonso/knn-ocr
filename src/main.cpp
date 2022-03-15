#include <iostream>
#include <fstream>
#include "pca.h"
#include "eigen.h"
#include "knn.h"

using namespace std;

void vector2kagglecsv(const Vector &v, const string &path) {
    cout << "Guardando resultado en: " << path << endl;
    /* Convierte la salida a csv para subir a kaggle */
    ofstream file(path);
    file << "ImageId,Label" << endl;
    for (int i = 0; i < v.rows(); i++) {
        file << i + 1 << "," << v.row(i) << endl;
    }
}

Matrix csv2matrix(const string &path) {
    std::cout << "Leyendo la matriz del archivo: " << path << std::endl;
    /* Convierte el csv a MatrixXD de Eigen
        https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix
    */
    ifstream indata;
    indata.open(path);
    string line;
    vector<double> values;
    uint rows = 0;
    //Ignoramos la linea que contiene los headers
    getline(indata, line);
    while (getline(indata, line)) {
        std::stringstream lineStream(line);
        string cell;
        while (getline(lineStream, cell, ',')) {
            values.push_back(stod(cell));
        }
        ++rows;
    }
    return Eigen::Map<Matrix>(values.data(), rows, values.size() / rows);

}

void run(const string &train_set_file, const string &test_set_file, const string &output_file, unsigned int k,
         unsigned int a, unsigned int method) {
    string path = "../data/";
    Matrix train_set_matrix = csv2matrix(path + train_set_file);
    Matrix y_train = train_set_matrix.col(0);
    Matrix X_train = train_set_matrix.block(0, 1, train_set_matrix.rows(), train_set_matrix.cols() - 1);
    Matrix X_predict = csv2matrix(path + test_set_file);

    KNNClassifier knn = KNNClassifier(k);

    const string votation_method = "distance";

    if (method == 0) { // kNN
        knn.fit(X_train, y_train);
        Vector y_predict = knn.predict(X_predict, votation_method);
        vector2kagglecsv(y_predict, path + "output/" + output_file);

    } else { // PCA + kNN
        PCA pca = PCA(a);
        pca.fit(X_train);
        Matrix X_train_trans = pca.transform(X_train);
        Matrix X_predict_trans = pca.transform(X_predict);

        knn.fit(X_train_trans, y_train);
        Vector y_predict = knn.predict(X_predict_trans, votation_method);
        vector2kagglecsv(y_predict, path + output_file);
    }
}

int main(int argc, char **argv) {
    /*
    Procesa los argumentos
    ./tp2 -m <method> --k <kNN parameter> --a <pca parameter> -i <training input rute> -t <test input rute> -o <output rute> 
    -k default 10
    -a default 30
    */
    //Chequear que tenga la cantidad de parámetros necesaria
    assert((argc == 9 || argc == 11 || argc == 13) && "Parámetros inválidos\n");

    string train_set_file, test_set_file, output_file;
    unsigned int k = 10;
    unsigned int a = 30;
    unsigned int method = atoi(argv[2]);

    //Setear los parámetros según el input
    if(argc == 9){
        train_set_file = argv[4];
        test_set_file = argv[6];
        output_file = argv[8];

    }else if(argc == 11){
        string optional = argv[3];
        assert((optional == "--k" || optional == "--a") && "Parámetros inválidos\n");

        train_set_file = argv[6];
        test_set_file = argv[8];
        output_file = argv[10];

        if (optional == "--k") {
            k = atoi(argv[4]);
            a = 30;
        } else if (optional == "--a") {
            k = 10;
            a = atoi(argv[4]);
        }

    }else if(argc == 13){
        train_set_file = argv[8];
        test_set_file = argv[10];
        output_file = argv[12];

        string optional1 = argv[3];
        string optional2 = argv[5];
        if (optional1 == "--k" && optional2 == "--a") {
            k = atoi(argv[4]);
            a = atoi(argv[6]);

        } else if (optional1 == "--a" && optional2 == "--k") {
            k = atoi(argv[6]);
            a = atoi(argv[4]);
        } else {
            printf("Párametros inválidos\n");
            return 1;
        }
    }

    run(train_set_file, test_set_file, output_file, k, a, method);

    return 0;
}
