//
// Created by Денис Марценюк on 19.03.2021.
//

#ifndef COMPARISON_OF_MULTIPLICATIONS_MATRIXOPERATIONS_H
#define COMPARISON_OF_MULTIPLICATIONS_MATRIXOPERATIONS_H

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cmath>

class MatrixOperations {

public:
    MatrixOperations();

    template<class T>
            static void generateMatrix(T *matrix, int rows, int columns);
    template<class T>
            static bool compareMatrices(T *matrixA, T *matrixB, int rows, int columns);
    template<class T>
            static void printMatrix(T *matrix, int rows, int columns);

};

template<class T>
void MatrixOperations::generateMatrix(T *matrix, int rows, int columns) {
    srand(time(0));
    int counter = 0;
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < columns; j++) {
            matrix[i * columns + j] = rand() / (T)RAND_MAX;
//            if (matrix[i * columns + j] < - 1.0) {
//                matrix[i * columns + j] = 1;
//            }
//            matrix[i * columns + j] = counter; //todo  проверяй всегда
            counter++;
        }
    }
}

template<class T>
bool MatrixOperations::compareMatrices(T *matrixA, T *matrixB, int rows, int columns) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < columns; j++) {
            if (abs(matrixA[i * columns + j] - matrixB[i * columns + j]) > 0.01) {
//                std::cout << "True: " << matrixA[i * columns + j] << " Nottrue " << matrixB[i * columns + j] << std::endl;
                return false;
            }
        }
    }
    return true;
}

template<class T>
void MatrixOperations::printMatrix(T *matrix, int rows, int columns) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < columns; j++) {
            std::cout << matrix[i * columns + j] << " ";
        }
        std::cout << std::endl;
    }
}


#endif //COMPARISON_OF_MULTIPLICATIONS_MATRIXOPERATIONS_H
