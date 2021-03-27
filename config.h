//
// Created by Денис Марценюк on 19.03.2021.
//

#ifndef COMPARISON_OF_MULTIPLICATIONS_CONFIG_H
#define COMPARISON_OF_MULTIPLICATIONS_CONFIG_H

#define NUMBER_OF_MEASUREMENTS 10

#define ROWS 16
#define COLUMNS 16

#define DEVICE_NUMBER 2
#define PATH_TO_KERNEL_FILE "/Users/mega_user/Desktop/GPU /comparison_of_transpose/kernels/transpose.cl"
#define WORK_GROUP_ROWS 16
#define WORK_GROUP_COLUMNS 16
#define THEORETICAL_PERFORMANCE_FLOAT 1.306
#define THEORETICAL_PERFORMANCE_DOUBLE 81.60

#define NAME_KERNEL_1 "simple_transpose_float"
#define TYPE_KERNEL_1 "float"
#define NAME_KERNEL_2 "simple_transpose_float"
#define TYPE_KERNEL_2 "float"
#define NAME_KERNEL_3 "simple_transpose_double"
#define TYPE_KERNEL_3 "double"
#define NAME_KERNEL_4 "optimization_1_multiplication_double"
#define TYPE_KERNEL_4 "double"

#endif //COMPARISON_OF_MULTIPLICATIONS_CONFIG_H
