#define BLOCK_SIZE 16

__kernel void simple_transpose_float( __global float* matrix, __global float* result, const int rows, const int columns) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < columns && y < rows) {
        result[x * rows + y] = matrix[y * columns + x];
    }
}

__kernel void simple_transpose_double( __global double* matrix, __global double* result, const int rows, const int columns) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < columns && y < rows) {
        result[x * rows + y] = matrix[y * columns + x];
    }
}

__kernel void optimization_1_transpose_float(__global const float* matrix, __global float* result, const int rows, const int cols) {
	int col = get_global_id(0);
	int row = get_global_id(1);

	int local_col = get_local_id(0);
	int local_row = get_local_id(1);

	__local float buffer[BLOCK_SIZE][BLOCK_SIZE];

	buffer[local_row][local_col] = matrix[row * cols + col];

	barrier(CLK_LOCAL_MEM_FENCE);

	result[row * rows + col] = buffer[local_col][local_row];
}

__kernel void optimization_1_transpose_double(__global const double* matrix, __global double* result, const int rows, const int cols) {
	int col = get_global_id(0);
	int row = get_global_id(1);

	int local_col = get_local_id(0);
	int local_row = get_local_id(1);

	__local double buffer[BLOCK_SIZE][BLOCK_SIZE];

	buffer[local_row][local_col] = matrix[row * cols + col];

	barrier(CLK_LOCAL_MEM_FENCE);

	result[row * rows + col] = buffer[local_col][local_row];
}