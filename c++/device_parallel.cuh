#ifndef MPRES_MP_VECTOR_CUH
#define MPRES_MP_VECTOR_CUH

#include "rns.cuh"
#include "multiprec.cuh"
#include "mpfloat.cuh"

// СТРУКТУРЫ

/*!
 * Буфер для векторного сложения
 */
typedef struct {
    int gamma;
    int theta;
    int factor_x;
    int factor_y;
} mp_add_vec_buf;


// МЕТОДЫ

namespace cuda {

    /*!
     * Kernel 1 - Параллельное умножение векторов: вычисление скалярных полей (знак, ИПХ, экспонента). Проверка необходимости округления.
     * @param result - вектр в глобальной памяти GPU
     * @param incr - расстояние между элементами result
     * @param x - вектр в глобальной памяти GPU
     * @param incx - расстояние между элементами x
     * @param y - вектр в глобальной памяти GPU
     * @param incy - расстояние между элементами y
     * @param n - размер операции
     */
    __global__ void mp_mul_vec_fields(mp_float_ptr result, int incr, mp_float_srcptr x, int incx, mp_float_srcptr y, int incy, unsigned int n) {
        unsigned int iterationsCount = (n / (gridDim.x * blockDim.x)) + 1;
        unsigned int i = 0;
        int numberIdx =  blockDim.x * blockIdx.x + threadIdx.x;
        int numberIdxX = numberIdx * incx;
        int numberIdxY = numberIdx * incy;
        int numberIdxR = numberIdx * incr;

        while (i < iterationsCount && numberIdx < n) {
            result[numberIdxR].sign = x[numberIdxX].sign ^ y[numberIdxY].sign;
            result[numberIdxR].exp = x[numberIdxX].exp + y[numberIdxY].exp;
            cuda::er_md_rd(&result[numberIdxR].ifc[0], &x[numberIdxX].ifc[0], &y[numberIdxY].ifc[0], &cuda::IFC_INVERSE_M.upp);
            cuda::er_md_ru(&result[numberIdxR].ifc[1], &x[numberIdxX].ifc[1], &y[numberIdxY].ifc[1], &cuda::IFC_INVERSE_M.low);
            //Переход к следующей итерации
            i += 1;
            numberIdx =  blockDim.x * (gridDim.x * i + blockIdx.x) + threadIdx.x;
            numberIdxX = numberIdx * incx;
            numberIdxY = numberIdx * incy;
            numberIdxR = numberIdx * incr;
        }
    }

    /*!
     * Kernel 2 - Параллельное умножение векторов: умножение остатков параллельно по модулям
     * @param result - вектр в глобальной памяти GPU
     * @param incr - расстояние между элементами result
     * @param x - вектор в глобальной памяти GPU
     * @param incx - расстояние между элементами x
     * @param y - вектор в глобальной памяти GPU
     * @param incy - расстояние между элементами y
     * @param n - размер операции
     */
    __global__ void mp_mul_vec_residues(mp_float_ptr result, int incr, mp_float_srcptr x, int incx, mp_float_srcptr y, int incy, unsigned int n) {

        int modulus = cuda::RNS_MODULI[threadIdx.x];

        unsigned int iterationsCount = (n / gridDim.x) + 1;
        unsigned int i = 0;
        unsigned int numberIdx = blockIdx.x;
        unsigned int numberIdxX = numberIdx * incx;
        unsigned int numberIdxY = numberIdx * incy;
        unsigned int numberIdxR = numberIdx * incr;
        unsigned int residueIdx = threadIdx.x;

        while (i < iterationsCount && numberIdx < n) {
            result[numberIdxR].residue[residueIdx] = (x[numberIdxX].residue[residueIdx] * y[numberIdxY].residue[residueIdx]) % modulus;
            i += 1;
            numberIdx  = gridDim.x * i + blockIdx.x;
            numberIdxX = numberIdx * incx;
            numberIdxY = numberIdx * incy;
            numberIdxR = numberIdx * incr;
        }
    }

    /**
     * Kernel 3 - Округление вектора чисел в глобальной памяти GPU. Метод одинаковый для умножения и сложения
     * Метод ускоренный. Для корректного округления с контроллируемой погрешностью см. соответствующую статью.
     * @param result - вектр в глобальной памяти GPU
     * @param incr - расстояние между элементами result
     * @param n - размер операции
     */
    __global__ void mp_vec_round(mp_float_ptr result, int incr, unsigned int n) {
        unsigned int iterationsCount =  (n / gridDim.x) + 1;
        unsigned int i = 0;
        int numberIdx = blockIdx.x;
        int numberIdxR = numberIdx * incr;

        while (i < iterationsCount && numberIdx < n) {
            int bits = (result[numberIdxR].ifc[1].exp - cuda::MP_H + 1)*(result[numberIdxR].ifc[1].frac != 0);
            while(bits > 0){
                //Масштабирование степенью двойки - НЕ РАБОТАЕТ ПРАВИЛЬНО.
                cuda::rns_scale2pow_thread(result[numberIdxR].residue, result[numberIdxR].residue, bits);
                //cuda::TEST_RNS_SCALE2POW_THREAD(result[numberIdxR].residue, result[numberIdxR].residue, bits);

                //Вычисление ИПХ (после округления число будет маленьким - верхняя граница ИПХ вычислится корректно, можно использовать fast-метод)
                if (threadIdx.x == 0) {
                    //TODO: потоковое округление работает неправильно.
                    //cuda::rns_scale2pow(result[numberIdxR].residue, result[numberIdxR].residue, bits);
                    result[numberIdxR].exp += bits;
                    //TODO: вместо того что ниже сделать новый метод ifc_compute для er_static_t ifc[2] и использовать потоковый метод вычисления ИПХ
                    ifc_static_t ifc;
                    cuda::ifc_compute_fast(&ifc, result[numberIdx].residue);
                    result[numberIdxR].ifc[0] = ifc.low;
                    result[numberIdxR].ifc[1] = ifc.upp;
                }
                bits = -1;
            }
            i += 1;
            numberIdx = gridDim.x * i + blockIdx.x;
            numberIdxR = numberIdx * incr;
            __syncthreads();
        }
     }


    /*!
     * Параллельное сложение векторов: вычисление скалярных полей (знак, ИПХ, экспонента, дополнительные поля)
     * @param result - вектр в глобальной памяти GPU
     * @param incr - расстояние между элементами вектора result TODO: необходимо реализовать
     * @param x - вектр в глобальной памяти GPU
     * @param incx - расстояние между элементами x
     * @param y - вектр в глобальной памяти GPU
     * @param incy - расстояние между элементами y
     * @param n - размер операции
     * @param buffer - буфер в глобальной памяти GPU для хранения временных переменных
     */
    __global__ void mp_add_vec_fields(mp_float_ptr result, int incr, mp_float_ptr x, int incx, mp_float_ptr y, int incy, unsigned int n, mp_add_vec_buf * buffer){
        unsigned int iterationsCount = (n / (gridDim.x * blockDim.x)) + 1;  // cuda::get_iterations_count_for_single_number_computations(n);
        unsigned int i = 0;
        int numberIdx =  blockDim.x * blockIdx.x + threadIdx.x;     // cuda::get_current_number_idx_for_single_number_computations(i);
        int numberIdxX = numberIdx * incx;                          // cuda::get_current_number_idx_for_single_number_computations(i, incx);
        int numberIdxY = numberIdx * incy;                          // cuda::get_current_number_idx_for_single_number_computations(i, incy);

        er_static_t ifc_x[2];
        er_static_t ifc_y[2];
        int exp_x;
        int exp_y;
        int sign_x;
        int sign_y;

        while (i < iterationsCount && numberIdx < n) {
            ifc_x[0] = x[numberIdxX].ifc[0];
            ifc_x[1] = x[numberIdxX].ifc[1];
            ifc_y[0] = y[numberIdxY].ifc[0];
            ifc_y[1] = y[numberIdxY].ifc[1];

            exp_x = x[numberIdxX].exp;
            exp_y = y[numberIdxY].exp;
            sign_x = x[numberIdxX].sign;
            sign_y = y[numberIdxY].sign;

            //Выравнивание экспонент
            int dexp = exp_x - exp_y;
            int gamma =  dexp * (dexp > 0); // если dexp > 0, то gamma =  dexp, иначе gamma = 0
            int theta = -dexp * (dexp < 0);  // если dexp < 0, то theta = -dexp, иначе theta = 0

            int nzx = ((ifc_y[1].frac == 0) || (theta + ifc_y[1].exp) < cuda::MP_J); // число u из статьи, оно = 1, если число X обнулять не нужно, иначе 0
            int nzy = ((ifc_x[1].frac == 0) || (gamma + ifc_x[1].exp) < cuda::MP_J); // число v из статьи, оно = 1, если число Y обнулять не нужно, иначе 0.

            gamma = gamma * nzy; //Если nzy = 0, т.е. число Y нужно обнулять, то gamma = 0, т.е. мы будем умножать число x на 2^0 фактически не меняя его
            theta = theta  * nzx; //Если nzx = 0, т.е. число X нужно обнулять, то theta = 0, т.е. мы будем умножать число y на 2^0 фактически не меняя его

            //Корректируем экспоненты
            exp_x = (exp_x - gamma) * nzx; // 0 если число x нужно обнулять
            exp_y = (exp_y - theta) * nzy; // 0 если число y нужно обнулять

            //Корректируем знаки
            sign_x *= nzx;
            sign_y *= nzy;

            int factor_x = (1 - 2 * sign_x) * nzx; // -1 если  x - отрицательное, 1 если x - положительное, 0 если x нужно обнулять (экспонента x очень маленькая)
            int factor_y = (1 - 2 * sign_y) * nzy; // -1 если  y - отрицательное, 1 если y - положительное, 0 если y нужно обнулять (экспонента y очень маленькая)

            //Умножаем ИПХ на степень двойки, восстанавливая верхние границы
            ifc_x[0].exp += gamma;
            ifc_x[1].exp += gamma;
            ifc_y[0].exp += theta;
            ifc_y[1].exp += theta;

            //Меняем знаки границ ИПХ, если слагаемые отрицательные (знак не изменится, если число положительное). ИПХ обнулится, если число надо занулять
            ifc_x[0].frac *=  factor_x;
            ifc_x[1].frac *=  factor_x;
            ifc_y[0].frac *=  factor_y;
            ifc_y[1].frac *=  factor_y;

            //Вычисление ИПХ суммы
            cuda::er_add_rd(&result[numberIdx].ifc[0], &ifc_x[sign_x], &ifc_y[sign_y]);
            cuda::er_add_ru(&result[numberIdx].ifc[1], &ifc_x[1 - sign_x], &ifc_y[1 - sign_y]);

            //Вычисление экспоненты и предварительное вычисление знака (знак может измениться когда требуется корректировка)
            result[numberIdx].sign = 0;
            result[numberIdx].exp = (exp_x == 0) ? exp_y : exp_x;

            //Корректировка отрицательного результата
            int plus  = result[numberIdx].ifc[0].frac >= 0 && result[numberIdx].ifc[1].frac >= 0; // 1 если результат положительный, 0 если результат отрицательный или неоднозначный случай
            int minus = result[numberIdx].ifc[0].frac < 0 && result[numberIdx].ifc[1].frac < 0;   // 1 если результат отрицательный, 0 если результат положительный или неоднозначный случай

            if(minus){
                result[numberIdx].sign = 1;
                er_static_t tmp = result[numberIdx].ifc[0];
                result[numberIdx].ifc[0].frac = -1 * result[numberIdx].ifc[1].frac;
                result[numberIdx].ifc[0].exp  = result[numberIdx].ifc[1].exp;
                result[numberIdx].ifc[1].frac = -1 * tmp.frac;
                result[numberIdx].ifc[1].exp  = tmp.exp;
            }

            if(!plus && !minus){
                printf("\n [WARNING]: Ambiguous number sign \n");
            }

            //Записываем в глобальный буфер переменные, необходимые для правильного вычисления мантиссы результата
            buffer[numberIdx].gamma = gamma;
            buffer[numberIdx].theta = theta;
            buffer[numberIdx].factor_x = factor_x;
            buffer[numberIdx].factor_y = factor_y;

            //Переходим к следующей итерации
            i += 1;
            numberIdx =  blockDim.x * (gridDim.x * i + blockIdx.x) + threadIdx.x;
            numberIdxX = numberIdx * incx;
            numberIdxY = numberIdx * incy;
        }


    }

    /*!
     * Параллельное сложение векторов: сложение остатков параллельно по модулям и восстановление (restoring) отрицательного результата
     * @param result - вектр в глобальной памяти GPU
     * @param incr - расстояние между элементами вектора result TODO: необходимо реализовать
     * @param x - вектр в глобальной памяти GPU
     * @param incx - расстояние между элементами x
     * @param y - вектр в глобальной памяти GPU
     * @param incy - расстояние между элементами y
     * @param n - размер операции
     * @param buffer - буфер в глобальной памяти GPU для хранения временных переменных
     */
    __global__ void mp_add_vec_residues(mp_float_ptr result, int incr, mp_float_ptr x, int incx, mp_float_ptr y, int incy, unsigned int n, mp_add_vec_buf * buffer){
        unsigned int iterationsCount = (n / gridDim.x) + 1;
        unsigned int i = 0;
        unsigned int numberIdx = blockIdx.x;
        unsigned int numberIdxX = numberIdx * incx;
        unsigned int numberIdxY = numberIdx * incy;
        unsigned int residueIdx = threadIdx.x;
        int modulus = cuda::RNS_MODULI[residueIdx];

        while(i < iterationsCount && numberIdx < n) {
            mp_add_vec_buf buf = buffer[numberIdx];
            int residue = (buf.factor_x * x[numberIdxX].residue[residueIdx] * cuda::RNS_POW2[buf.gamma][residueIdx] +
                           buf.factor_y * y[numberIdxY].residue[residueIdx] * cuda::RNS_POW2[buf.theta][residueIdx]) % modulus;
            if(result[numberIdx].sign == 1){
                residue = (modulus - residue) % modulus;
            }
            result[numberIdx].residue[residueIdx] = residue < 0 ? residue + modulus : residue;

            i += 1;
            numberIdx  = gridDim.x * i + blockIdx.x;
            numberIdxX = numberIdx * incx;
            numberIdxY = numberIdx * incy;
        }
    }
}

#endif //MPRES_VECTOR_CUH


/**
 * BLAS подпрограмма ASUM
 */

#ifndef MF_ASUM
#define MF_ASUM


#include "multiprec.cuh"
#include "smem.cuh"
#include "mpfloat.cuh"

namespace cuda {

    /*
     * Ядро редукции на GPU
     */
    static __global__ void mp_asum_kernel(int n, mp_float_ptr g_idata, int incx, mp_float_ptr g_odata) {

        mp_static_t * sdata = smem<mp_static_t>();

        // parameters
        unsigned int tid = threadIdx.x;
        unsigned int bid = blockIdx.x;
        unsigned int bsize = blockDim.x;

        unsigned int i = bid * bsize * 2 + tid;
        unsigned int k = 2 * gridDim.x * bsize;

        // perform first level of reduction,
        // reading from global memory, writing to shared memory
        sdata[tid] = cuda::MP_ZERO;

        // we reduce multiple elements per thread.  The number is determined by the
        // number of active thread blocks (via gridDim).  More blocks will result
        // in a larger gridSize and therefore fewer elements per thread
        int gind = i * incx;
        int hind = (i + bsize) * incx; // half
        while (i < n) {

            cuda::mp_add_abs2(&sdata[tid], &sdata[tid], &g_idata[gind]);
            // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
            if (i + bsize < n){
                cuda::mp_add_abs2(&sdata[tid], &sdata[tid], &g_idata[hind]);
            }
            i += k;
            gind = i * incx;
            hind = (i + bsize) * incx;
        }

        __syncthreads();

        i = bsize;
        while(i >= 2){
            unsigned int half = i >> 1;
            hind = (tid+half);
            if ((bsize >= i) && (tid < half)) {
                cuda::mp_add_abs2(&sdata[tid], &sdata[tid], &sdata[hind]);
            }
            i = i >> 1;
            __syncthreads();
        }
        // write result for this block to global mem
        if (tid == 0) {
            g_odata[bid] = sdata[tid];
        };
    }
}

/*
 * Остаточная редукция на CPU
 * Не используется, так как границы ошибок для попарной редукции все равно не держатся
 */
void pairwise_abs_reduce(mp_float_ptr result, mp_float_ptr buffer, int n){
    mp_static_t v;
    int p = 0;
    mp_static_t stack[10];
    for(int i = 0; i < n; ++i){
        v = buffer[i]; // shift
        for(size_t b = 1; i & b; b <<= 1, --p) //reduce
            mp_add_abs2(&v, &v, &stack[p-1]);
        stack[p++] = v;
    }

    *result = *MP_ZERO;

    while (p){
        mp_add_abs2(result, result, &stack[--p]);
    }
}


/*!
 * Вычисление asum с гибридной редукцией. Возвращает результат в указателе на память хоста (result)
 * @tparam gridDim   - Число параллельных блоков для редукции
 * @tparam blockDim  - Размер блока для редукции
 */
template <int gridDim1, int blockDim1>
void mp_masum(int n, mp_float_ptr x, int incx, mp_float_ptr result){

    mp_float_ptr d_buf; // device buffer

    //Allocate memory buffers for the device results
    cudaMalloc((void **) &d_buf, sizeof(mp_static_t) * gridDim1);

    //Compute the size of shared memory allocated per block
    int smemsize1 = (blockDim1 + 1) * sizeof(mp_static_t);
    int smemsize2 = (gridDim1 + 1) * sizeof(mp_static_t);

    //Launch the CUDA kernel to perform parallel summation on the GPU
    cuda::mp_asum_kernel <<< gridDim1, blockDim1, smemsize1>>> (n, x, incx, d_buf);
    //Launch the CUDA kernel to perform summation of the results of parallel blocks on the GPU
    cuda::mp_asum_kernel <<< 1, gridDim1, smemsize2>>> (gridDim1, d_buf, incx, result);

    //Copy the sum into the host memory
    //cudaMemcpy(result, d_buf, sizeof(mp_static_t), cudaMemcpyDeviceToHost);
    //*result = d_buf[0];
    /*
        //Copy the results of the blocks
        cudaMemcpy(h_buf, d_buf, sizeof(mp_static_t) * G, cudaMemcpyDeviceToHost);
        //CPU reduction
        for (int i = 0; i < G; i++) {
            ::mp_add_abs2(result, result, &h_buf[i]);
        }
    */
    
    // Cleanup
    cudaFree(d_buf);
}

#endif
