//
// Created by Денис Марценюк on 19.03.2021.
//

#ifndef COMPARISON_OF_MULTIPLICATIONS_CONFIG_H
#define COMPARISON_OF_MULTIPLICATIONS_CONFIG_H

#define NUMBER_OF_MEASUREMENTS 10

#define ROWS 4096
#define COLUMNS 4096

#define DEVICE_NUMBER 2
#define PATH_TO_KERNEL_FILE "/Users/mega_user/Desktop/GPU /comparison_of_transpose/kernels/transpose.cl"
#define WORK_GROUP_ROWS 16
#define WORK_GROUP_COLUMNS 16
#define THROUGHTPUT_MEMORY 81.6

#define NAME_KERNEL_1 "simple_transpose_float"
#define TYPE_KERNEL_1 "float"
#define NAME_KERNEL_2 "optimization_1_transpose_float"
#define TYPE_KERNEL_2 "float"
#define NAME_KERNEL_3 "simple_transpose_double"
#define TYPE_KERNEL_3 "double"
#define NAME_KERNEL_4 "optimization_1_transpose_double"
#define TYPE_KERNEL_4 "double"

#endif //COMPARISON_OF_MULTIPLICATIONS_CONFIG_H
