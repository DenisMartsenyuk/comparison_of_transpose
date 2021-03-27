__kernel void simple_transpose_float( __global float* matrix, __global float* result, const int rows, const int columns) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= columns || row >= rows) {
        return;
    }
    float element = matrix[row * columns + col];
    result[col * rows + row] = element;
}

__kernel void simple_transpose_double( __global double* matrix, __global double* result, const int rows, const int columns) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= columns || row >= rows) {
        return;
    }
    float element = matrix[row * columns + col];
    result[col * rows + row] = element;
}