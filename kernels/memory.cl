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
    float res;
    if (x < columns && y < rows) {
        res = matrix[y * columns + x];
        if (res < -1.0) {
            matrix[y * columns + x] = -1.0;
        }
    }
}

__kernel void read_double( __global double* matrix, __global double* result, const int rows, const int columns) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    double res;
    if (x < columns && y < rows) {
        res = matrix[y * columns + x];
        if (res < -1.0) {
            matrix[y * columns + x] = -1.0;
        }
    }
}

__kernel void copy_float( __global float* matrix, __global float* result, const int rows, const int columns) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < columns && y < rows) {
        result[y * columns + x] = matrix[y * columns + x];
    }
}

__kernel void copy_double( __global double* matrix, __global double* result, const int rows, const int columns) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < columns && y < rows) {
        result[y * columns + x] = matrix[y * columns + x];
    }
}

__kernel void copy_array_float( __global float* matrix, __global float* result, const int rows, const int columns) {
    int x = get_global_id(0);
    if (x < columns * rows) {
        result[x] = matrix[x];
    }
}

__kernel void copy_array_double( __global double* matrix, __global double* result, const int rows, const int columns) {
    int x = get_global_id(0);
    if (x < columns * rows) {
        result[x] = matrix[x];
    }
}