//
// Created by Денис Марценюк on 27.03.2021.
//

#ifndef COMPARISON_OF_TRANSPOSE_TRANSPOSEMATRICESGPU_H
#define COMPARISON_OF_TRANSPOSE_TRANSPOSEMATRICESGPU_H

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include "libraries/cl2.hpp"
#include <fstream>

class TransposeMatricesGPU {
private:
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::NDRange workItems;
    cl::NDRange workGroup;
    cl::Kernel kernel;
    cl::CommandQueue queue;
    cl::Buffer bufferMatrix;
    cl::Buffer bufferB;
    cl::Buffer bufferResult;
    cl::Event event;

    std::string readKernel(std::string path);

    template<class T>
    void createBuffersAndQueue(int rows, int columns);

public:
    int init(int deviceNumber);
    void setProgram(std::string path);
    void setWorkGroupAndWorkItems(int rowsWorkGroup, int columnsWorkGroup, int rowsMatrix, int columnsMatrix);
    void setKernel(std::string nameKernel);

    template<class T>
    void setArgs(T *matrix, int rows, int columns);

    void executeKernel();
    double getExecutionTime();

    template<class T>
    void getResult(T *matrixResult, int rows, int columns);

};

template<class T>
void TransposeMatricesGPU::createBuffersAndQueue(int rows, int columns) {
    bufferMatrix = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(T) * rows * columns);
    bufferResult = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T) * rows * columns);
    queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
}

template<class T>
void TransposeMatricesGPU::setArgs(T *matrix, int rows,
                                   int columns) {
    createBuffersAndQueue<T>(rows, columns);
    queue.enqueueWriteBuffer(bufferMatrix, CL_TRUE, 0, sizeof(T) * rows * columns, &matrix[0]);
    queue.finish();
    kernel.setArg(0, bufferMatrix);
    kernel.setArg(1, bufferResult);
    kernel.setArg(2, sizeof(int), &rows);
    kernel.setArg(3, sizeof(int), &columns);

}

template<class T>
void TransposeMatricesGPU::getResult(T *matrixResult, int rows, int columns) {
    queue.enqueueReadBuffer(bufferResult, CL_TRUE, 0, sizeof(T) * rows * columns, &matrixResult[0]);
}

#endif //COMPARISON_OF_TRANSPOSE_TRANSPOSEMATRICESGPU_H
