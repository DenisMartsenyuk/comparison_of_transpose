__kernel void write_float( __global float* matrix, __global float* result, const int rows, const int columns) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < columns && y < rows) {
        result[y * columns + x] = 1.0;
    }
}

__kernel void write_double( __global double* matrix, __global double* result, const int rows, const int columns) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < columns && y < rows) {
        result[y * columns + x] = 1.0;
    }
}

__kernel void read_float( __global float* matrix, __global float* result, const int rows, const int columns) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < columns && y < rows) {
        if (matrix[y * columns + x] < -1.0) {
            matrix[y * columns + x] = -1.0;
        }
    }
}

__kernel void read_double( __global double* matrix, __global double* result, const int rows, const int columns) {
    int x = get_global_id(0);
        int y = get_global_id(1);
        if (x < columns && y < rows) {
            if (matrix[y * columns + x] < -1.0) {
                matrix[y * columns + x] = -1.0;
            }
        }
}