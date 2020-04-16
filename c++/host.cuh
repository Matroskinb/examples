#pragma once

#include "reduce.cuh"
#include "vectors.cuh"
#include "mp_vectors.cuh"

namespace cuda {

    /*!
     * Вычисление  axpy. Принимает, возвращает указатели на device
     * @tparam gridDim1  - Число параллельных блоков для расчета знака, экспоненты, ИПХ
     * @tparam blockDim1 - Размер блока для расчета знака, экспоненты, ИПХ
     * @tparam gridDim2  - Число параллельных блоков для умножения, сложения и округления
     */
    template <int gridDim1, int blockDim1, int gridDim2>
    void mp_maxpy(int n, mp_float_ptr x, int incx, mp_float_ptr y, int incy, mp_float_ptr alpha) {
        mp_float_ptr d_vector;
        mp_add_vec_buf * d_buffer;

        // Allocate memory for multiplication result
        cudaMalloc(&d_vector, sizeof(mp_static_t) * n);
        cudaMalloc(&d_buffer, sizeof(mp_add_vec_buf) * n);

        // Multiplication - Computing the signs, exponents, interval evaluations
        cuda::mp_mul_vec_fields <<< gridDim1, blockDim1 >>> (d_vector, 1, x, incx, alpha, 0, n);

        // Multiplication - Residue multiplication
        cuda::mp_mul_vec_residues <<< gridDim2, RNS_MODULI_SIZE >>> (d_vector, 1, x, incx, alpha, 0, n);

        // Multiplication - Rounding intermediate
        cuda::mp_vec_round <<< gridDim2, RNS_MODULI_SIZE >>> (d_vector, 1, n);

        // Summation - Computing the signs, exponents, interval evaluations
        cuda::mp_add_vec_fields <<< gridDim1, blockDim1 >>> (y, 1, d_vector, 1, y, 1, n, d_buffer);

        // Summation - Residue summation
        cuda::mp_add_vec_residues <<< gridDim2, RNS_MODULI_SIZE >>> (y, 1,  d_vector, 1, y, 1, n, d_buffer);

        // Final rounding
        cuda::mp_vec_round <<< gridDim2, RNS_MODULI_SIZE >>>(y, incy, n);

        // Cleanup
        cudaFree(d_vector);
        cudaFree(d_buffer);
    }
}