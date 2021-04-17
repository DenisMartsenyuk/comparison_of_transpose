#include <iostream>
#include "matrices/MatrixOperations.h"
#include "config.h"
#include "gpu/TransposeMatricesGPU.h"

template <class T>
double test(TransposeMatricesGPU *transposeMatricesGpu, std::pair<std::string, std::string> *kernel, T *matrix, T *resultReference, T *resultKernel) {
    transposeMatricesGpu->setKernel(kernel->second);
    transposeMatricesGpu->setArgs(matrix, ROWS, COLUMNS);
    transposeMatricesGpu->executeKernel();
    transposeMatricesGpu->getResult(resultKernel, ROWS, COLUMNS);
//    if (MatrixOperations::compareMatrices(resultReference, resultKernel, ROWS, COLUMNS)) {
        return transposeMatricesGpu->getExecutionTime();
//    }
//    return -1.0;
}

template <class T>
void createTestData(TransposeMatricesGPU *transposeMatricesGpu, std::string kernel, T *matrix, T *resultReference) {
    MatrixOperations::generateMatrix(matrix, ROWS, COLUMNS);
    transposeMatricesGpu->setKernel(kernel);
    transposeMatricesGpu->setArgs(matrix, ROWS, COLUMNS);
    transposeMatricesGpu->executeKernel();
    transposeMatricesGpu->getResult(resultReference, COLUMNS, ROWS);
//    MatrixOperations::printMatrix(resultReference, COLUMNS, ROWS);
//    std::cout << "Time: " << transposeMatricesGpu->getExecutionTime() << std::endl;
}

void printTestResult(int numberTest, std::vector<std::pair<std::string, std::vector<double>>> &resultTime) {
//    std::cout << "Test " << numberTest << std::endl;
    for (int i = 0; i < resultTime.size(); ++i) {
//        std::cout << "Times " << resultTime[i].first << " kernel: ";
        for (int j = 0; j < resultTime[i].second.size(); ++j) {
            std::cout << resultTime[i].second[j] << " ";
        }
//        std::cout << std::endl;
//        std::cout << "in milliseconds" << std::endl;
    }
    std::cout << std::endl;
}

int main() {

    double *matrixDouble = (double*)malloc(ROWS * COLUMNS * sizeof(double));
    double *resultDoubleReference = (double*)malloc(ROWS * COLUMNS * sizeof(double));
    double *resultDoubleKernel = (double*)malloc(ROWS * COLUMNS * sizeof(double));

    float *matrixFloat = (float*)malloc(ROWS * COLUMNS * sizeof(float));
    float *resultFloatReference = (float*)malloc(ROWS * COLUMNS * sizeof(float));
    float *resultFloatKernel = (float*)malloc(ROWS * COLUMNS * sizeof(float));

    std::vector<std::pair<std::string, std::string>> kernels;
    kernels.emplace_back(TYPE_KERNEL_1, NAME_KERNEL_1);
//    kernels.emplace_back(TYPE_KERNEL_2, NAME_KERNEL_2);
//    kernels.emplace_back(TYPE_KERNEL_3, NAME_KERNEL_3);
//    kernels.emplace_back(TYPE_KERNEL_4, NAME_KERNEL_4);

    TransposeMatricesGPU transposeMatricesGpu = TransposeMatricesGPU();
    transposeMatricesGpu.init(DEVICE_NUMBER);
    transposeMatricesGpu.setProgram(PATH_TO_KERNEL_FILE);
    transposeMatricesGpu.setWorkGroupAndWorkItems(WORK_GROUP_ROWS, WORK_GROUP_COLUMNS, ROWS, COLUMNS);

    for (int i = 0; i < NUMBER_OF_MEASUREMENTS; ++i) {
        std::vector<std::pair<std::string, std::vector<double>>> resultTime;
        resultTime.emplace_back(NAME_KERNEL_1, std::vector<double> {});
//        resultTime.emplace_back(NAME_KERNEL_2, std::vector<double> {});
//        resultTime.emplace_back(NAME_KERNEL_3, std::vector<double> {});
//        resultTime.emplace_back(NAME_KERNEL_4, std::vector<double> {});
        for (int j = 0; j < kernels.size(); ++j) {
            if (kernels[j].first == "double") {
                createTestData(&transposeMatricesGpu, NAME_REFERENCE_KERNEL_DOUBLE, matrixDouble, resultDoubleReference);
                for (int k = 0; k < NUMBER_OF_IDENTICAL_MEASUREMENTS; ++k) {
                    resultTime[j].second.push_back(test(&transposeMatricesGpu, &kernels[j], matrixDouble, resultDoubleReference, resultDoubleKernel));
                }
            } else if(kernels[j].first == "float") {
                createTestData(&transposeMatricesGpu, NAME_REFERENCE_KERNEL_FLOAT, matrixFloat, resultFloatReference);
                for (int k = 0; k < NUMBER_OF_IDENTICAL_MEASUREMENTS; ++k) {
                    resultTime[j].second.push_back(test(&transposeMatricesGpu, &kernels[j], matrixFloat, resultFloatReference, resultFloatKernel));
                }
            }
        }
        printTestResult(i + 1, resultTime);
    }

    return 0;
}
