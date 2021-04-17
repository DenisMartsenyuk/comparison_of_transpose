//
// Created by Денис Марценюк on 19.03.2021.
//

#ifndef COMPARISON_OF_MULTIPLICATIONS_CONFIG_H
#define COMPARISON_OF_MULTIPLICATIONS_CONFIG_H

#define NUMBER_OF_MEASUREMENTS 5
#define NUMBER_OF_IDENTICAL_MEASUREMENTS 10

#define ROWS 2048
#define COLUMNS 2048

#define DEVICE_NUMBER 2
#define PATH_TO_KERNEL_FILE "/Users/mega_user/Desktop/GPU /comparison_of_transpose/kernels/memory.cl" //transpose.cl
#define WORK_GROUP_ROWS 16
#define WORK_GROUP_COLUMNS 16

#define NAME_REFERENCE_KERNEL_FLOAT "copy_array_float" //simple_transpose_float
#define NAME_REFERENCE_KERNEL_DOUBLE "copy_array_double" //simple_transpose_double

#define NAME_KERNEL_1 "copy_array_float" //simple_transpose_float
#define TYPE_KERNEL_1 "float"
#define NAME_KERNEL_2 "optimization_1_transpose_float"
#define TYPE_KERNEL_2 "float"
#define NAME_KERNEL_3 "copy_array_double" //simple_transpose_double
#define TYPE_KERNEL_3 "double"
#define NAME_KERNEL_4 "optimization_1_transpose_double"
#define TYPE_KERNEL_4 "double"

#endif //COMPARISON_OF_MULTIPLICATIONS_CONFIG_H
