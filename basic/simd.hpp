#ifndef SIMD_HPP
#define SIMD_HPP
#include <immintrin.h>
#include <vector>
#include <cmath>

namespace simd {

struct SSE2 {

#if defined(__SSE2__)

    inline static void add(double* __restrict z, const double* __restrict y, const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        const double *py = y;
        double *pz = z;
        /* __m128d: SSE2 double */
        __m128d vecx;
        __m128d vecy;
        __m128d vecz;
        /* offset */
        std::size_t offset = sizeof (__m256)/sizeof (double);
        for (std::size_t i = 0; i < N/offset; i++) {
            /* put 2 double into __m128d */
            vecx = _mm_load_pd(px);
            vecy = _mm_load_pd(py);
            /* add */
            vecz = _mm_add_pd(vecx, vecy);
            /* store result */
            _mm_store_pd(pz, vecz);
            /* move */
            px += offset;
            py += offset;
            pz += offset;
        }
        return;
    }

    inline static void sub(double* __restrict z, const double* __restrict y, const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        const double *py = y;
        double *pz = z;
        /* __m128d: SSE2 double */
        __m128d vecx;
        __m128d vecy;
        __m128d vecz;
        /* offset */
        std::size_t offset = sizeof (__m128d)/sizeof (double);
        for (std::size_t i = 0; i < N/offset; i++) {
            /* put 2 double into __m128d */
            vecx = _mm_load_pd(px);
            vecy = _mm_load_pd(py);
            /* sub */
            vecz = _mm_sub_pd(vecx, vecy);
            /* store result */
            _mm_store_pd(pz, vecz);
            /* move */
            px += offset;
            py += offset;
            pz += offset;
        }
        return;
    }

    inline static void mul(double* __restrict z, const double* __restrict y, const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        const double *py = y;
        double *pz = z;
        /* __m128d: SSE2 double */
        __m128d vecx;
        __m128d vecy;
        __m128d vecz;
        /* offset */
        std::size_t offset = sizeof (__m128d)/sizeof (double);
        for (std::size_t i = 0; i < N/offset; i++) {
            /* put 2 double into __m128d */
            vecx = _mm_load_pd(px);
            vecy = _mm_load_pd(py);
            /* multipy */
            vecz = _mm_mul_pd(vecx, vecy);
            /* store result */
            _mm_store_pd(pz, vecz);
            /* move */
            px += offset;
            py += offset;
            pz += offset;
        }
        return;
    }

    inline static void div(double* __restrict z, const double* __restrict y, const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        const double *py = y;
        double *pz = z;
        /* __m128d: SSE2 double */
        __m128d vecx;
        __m128d vecy;
        __m128d vecz;
        /* offset */
        std::size_t offset = sizeof (__m128d)/sizeof (double);
        for (std::size_t i = 0; i < N/offset; i++) {
            /* put 2 double into __m128d */
            vecx = _mm_load_pd(px);
            vecy = _mm_load_pd(py);
            /* divide */
            vecz = _mm_div_pd(vecx, vecy);
            /* store result */
            _mm_store_pd(pz, vecz);
            /* move */
            px += offset;
            py += offset;
            pz += offset;
        }
        return;
    }

    inline static void add(float* __restrict z, const float* __restrict y, const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        const float *py = y;
        float *pz = z;
        /* __m128: SSE2 float */
        __m128 vecx;
        __m128 vecy;
        __m128 vecz;
        /* offset */
        std::size_t offset = sizeof (__m128)/sizeof (float);
        for (std::size_t i = 0; i < N/offset; i++) {
            /* put 4 float into __m128 */
            vecx = _mm_loadu_ps(px);
            vecy = _mm_loadu_ps(py);
            /* add */
            vecz = _mm_add_ps(vecx, vecy);
            /* store result */
            _mm_storeu_ps(pz, vecz);
            /* move */
            px += offset;
            py += offset;
            pz += offset;
        }
        return;
    }

    inline static void sub(float* __restrict z, const float* __restrict y, const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        const float *py = y;
        float *pz = z;
        /* __m128: SSE2 float */
        __m128 vecx;
        __m128 vecy;
        __m128 vecz;
        /* offset */
        std::size_t offset = sizeof (__m128)/sizeof (float);
        for (std::size_t i = 0; i < N/offset; i++) {
            /* put 4 float into __m128 */
            vecx = _mm_loadu_ps(px);
            vecy = _mm_loadu_ps(py);
            /* add */
            vecz = _mm_sub_ps(vecx, vecy);
            /* store result */
            _mm_storeu_ps(pz, vecz);
            /* move */
            px += offset;
            py += offset;
            pz += offset;
        }
        return;
    }

    inline static void mul(float* __restrict z, const float* __restrict y, const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        const float *py = y;
        float *pz = z;
        /* __m128: SSE2 float */
        __m128 vecx;
        __m128 vecy;
        __m128 vecz;
        /* offset */
        std::size_t offset = sizeof (__m128)/sizeof (float);
        for (std::size_t i = 0; i < N/offset; i++) {
            /* put 4 float into __m128 */
            vecx = _mm_loadu_ps(px);
            vecy = _mm_loadu_ps(py);
            /* mul */
            vecz = _mm_mul_ps(vecx, vecy);
            /* store result */
            _mm_storeu_ps(pz, vecz);
            /* move */
            px += offset;
            py += offset;
            pz += offset;
        }
        return;
    }

    inline static void div(float* __restrict z, const float* __restrict y, const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        const float *py = y;
        float *pz = z;
        /* __m128: SSE2 float */
        __m128 vecx;
        __m128 vecy;
        __m128 vecz;
        /* offset */
        std::size_t offset = sizeof (__m128)/sizeof (float);
        for (std::size_t i = 0; i < N/offset; i++) {
            /* put 4 float into __m128 */
            vecx = _mm_loadu_ps(px);
            vecy = _mm_loadu_ps(py);
            /* div */
            vecz = _mm_div_ps(vecx, vecy);
            /* store result */
            _mm_storeu_ps(pz, vecz);
            /* move */
            px += offset;
            py += offset;
            pz += offset;
        }
        return;
    }

    inline static void matMul2(double* __restrict z, std::size_t zRow, std::size_t zCol,
                               const double* __restrict x, std::size_t xRow, std::size_t xCol,
                               const double* __restrict y, std::size_t yRow, std::size_t yCol)
    {
        const double *x_ = x;
        const double *y_ = y;
        double *z_ = z;
        std::size_t r = zCol%2;
        for (std::size_t i = 0; i < zRow; i++) {
            for (std::size_t k = 0; k < xCol; k++) {
                double xik = x_[i*xCol + k];
                __m128d vecx = _mm_set1_pd(xik);
                for (std::size_t j = 0; j < zCol - r; j+=2) {
                    __m128d vecy = _mm_loadu_pd(y_ + k*yCol + j);
                    __m128d vecz = _mm_loadu_pd(z_ + i*xCol + j);
                    /* _mm256_fmadd_ps(a, b, c): a*b + c */
                    vecz = _mm_fmadd_pd(vecx, vecy, vecz);
                    /* store result */
                    _mm_store_pd(z_ + i*xCol + j, vecz);
                }
                for (std::size_t j = zCol - r; j < zCol; j++) {
                    z_[i*zCol + j] += xik * y_[k*yCol + j];
                }
            }
        }
        return;
    }

    inline static void matMul4(float* __restrict z, std::size_t zRow, std::size_t zCol,
                               const float* __restrict x, std::size_t xRow, std::size_t xCol,
                               const float* __restrict y, std::size_t yRow, std::size_t yCol)
    {
        const float *x_ = x;
        const float *y_ = y;
        float *z_ = z;
        std::size_t r = zCol%4;
        for (std::size_t i = 0; i < zRow; i++) {
            for (std::size_t k = 0; k < xCol; k++) {
                float xik = x_[i*xCol + k];
                __m128 vecx = _mm_set1_ps(xik);   
                for (std::size_t j = 0; j < zCol - r; j+=4) {
                    __m128 vecy = _mm_loadu_ps(y_ + k*yCol + j);
                    __m128 vecz = _mm_loadu_ps(z_ + i*xCol + j);
                    /* _mm_fmadd_ps(a, b, c): a*b + c */
                    vecz = _mm_fmadd_ps(vecx, vecy, vecz);
                    /* store result */
                    _mm_store_ps(z_ + i*xCol + j, vecz);
                }
                for (std::size_t j = zCol - r; j < zCol; j++) {
                    z_[i*zCol + j] += xik * y_[k*yCol + j];
                }
            }
        }
        return;
    }

#endif // SSE2
};

/*
   avx2 on windows:
        implement command "bcdedit/set xsavedisable 0" and set compiler option: "/arch:AVX2"
        https://devblogs.microsoft.com/cppblog/avx2-support-in-visual-studio-c-compiler/

   avx2 on linux: sudo apt-get install libmkl-dev libmkl-avx

*/

struct AVX2 {

#if defined(__AVX2__)

    inline static void add(double* __restrict z, const double* __restrict y, const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        const double *py = y;
        double *pz = z;
        /* __m256d: AVX double */
        __m256d vecx;
        __m256d vecy;
        __m256d vecz;
        /* step */
        std::size_t step = sizeof (__m256d)/sizeof (double);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            /* put 4 double into __m256d */
            vecx = _mm256_load_pd(px + i);
            vecy = _mm256_load_pd(py + i);
            /* add */
            vecz = _mm256_add_pd(vecx, vecy);
            /* store result */
            _mm256_store_pd(pz + i, vecz);
        }
        return;
    }

    inline static void sub(double* __restrict z, const double* __restrict y, const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        const double *py = y;
        double *pz = z;
        /* __m256d: AVX double */
        __m256d vecx;
        __m256d vecy;
        __m256d vecz;
        /* offset */
        /* step */
        std::size_t step = sizeof (__m256d)/sizeof (double);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            /* put 4 double into __m256d */
            vecx = _mm256_load_pd(px + i);
            vecy = _mm256_load_pd(py + i);
            /* add */
            vecz = _mm256_sub_pd(vecx, vecy);
            /* store result */
            _mm256_store_pd(pz + i, vecz);
        }
        return;
    }

    inline static void mul(double* __restrict z, const double* __restrict y, const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        const double *py = y;
        double *pz = z;
        /* __m256d: AVX double */
        __m256d vecx;
        __m256d vecy;
        __m256d vecz;
        /* step */
        std::size_t step = sizeof (__m256d)/sizeof (double);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            /* put 4 double into __m256d */
            vecx = _mm256_load_pd(px + i);
            vecy = _mm256_load_pd(py + i);
            /* add */
            vecz = _mm256_mul_pd(vecx, vecy);
            /* store result */
            _mm256_store_pd(pz + i, vecz);
        }
        return;
    }

    inline static void div(double* __restrict z, const double* __restrict y, const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        const double *py = y;
        double *pz = z;
        /* __m256d: AVX double */
        __m256d vecx;
        __m256d vecy;
        __m256d vecz;
        /* step */
        std::size_t step = sizeof (__m256d)/sizeof (double);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            /* put 4 double into __m256d */
            vecx = _mm256_load_pd(px + i);
            vecy = _mm256_load_pd(py + i);
            /* add */
            vecz = _mm256_div_pd(vecx, vecy);
            /* store result */
            _mm256_store_pd(pz + i, vecz);
        }
        return;
    }

    inline static void add(float* __restrict z, const float* __restrict y, const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        const float *py = y;
        float *pz = z;
        /* __m256: AVX2 float */
        __m256 vecx;
        __m256 vecy;
        __m256 vecz;
        /* step */
        std::size_t step = sizeof (__m256)/sizeof (float);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            /* put 4 double into __m256d */
            vecx = _mm256_load_ps(px + i);
            vecy = _mm256_load_ps(py + i);
            /* add */
            vecz = _mm256_add_ps(vecx, vecy);
            /* store result */
            _mm256_store_ps(pz + i, vecz);
        }
        return;
    }

    inline static void sub(float* __restrict z, const float* __restrict y, const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        const float *py = y;
        float *pz = z;
        /* __m256: AVX2 float */
        __m256 vecx;
        __m256 vecy;
        __m256 vecz;
        /* step */
        std::size_t step = sizeof (__m256)/sizeof (float);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            /* put 4 double into __m256d */
            vecx = _mm256_load_ps(px + i);
            vecy = _mm256_load_ps(py + i);
            /* add */
            vecz = _mm256_sub_ps(vecx, vecy);
            /* store result */
            _mm256_store_ps(pz + i, vecz);
        }
        return;
    }

    inline static void mul(float* __restrict z, const float* __restrict y, const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        const float *py = y;
        float *pz = z;
        /* __m256: AVX2 float */
        __m256 vecx;
        __m256 vecy;
        __m256 vecz;
        /* step */
        std::size_t step = sizeof (__m256)/sizeof (float);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            /* put 4 double into __m256d */
            vecx = _mm256_load_ps(px + i);
            vecy = _mm256_load_ps(py + i);
            /* add */
            vecz = _mm256_mul_ps(vecx, vecy);
            /* store result */
            _mm256_store_ps(pz + i, vecz);
        }
        return;
    }

    inline static void div(float* __restrict z, const float* __restrict y, const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        const float *py = y;
        float *pz = z;
        /* __m256: AVX2 float */
        __m256 vecx;
        __m256 vecy;
        __m256 vecz;
        /* step */
        std::size_t step = sizeof (__m256)/sizeof (float);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            /* put 4 double into __m256d */
            vecx = _mm256_load_ps(px + i);
            vecy = _mm256_load_ps(py + i);
            /* add */
            vecz = _mm256_div_ps(vecx, vecy);
            /* store result */
            _mm256_store_ps(pz + i, vecz);
        }
        return;
    }

    inline static double max(const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        /* align */
        std::size_t r = N%4;
        /* find max value per 8 element */
        __m256d maxVec = _mm256_loadu_pd(px);
        for (std::size_t i = 4; i < N - r; i+=4) {
            __m256d vecx = _mm256_loadu_pd(px + i);
            maxVec = _mm256_max_pd(vecx, maxVec);
        }
        /* find max value in result */
        double result[4];
        _mm256_store_pd(result, maxVec);
        double maxValue = result[0];
        for (std::size_t i = 1; i < 4; i++) {
            maxValue = result[i] > maxValue ? result[i] : maxValue;
        }
        /* find max value in the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            maxValue = px[i] > maxValue ? px[i] : maxValue;
        }
        return maxValue;
    }

    inline static float max(const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        /* align */
        std::size_t r = N%8;
        /* find max value per 8 element */
        __m256 maxVec = _mm256_loadu_ps(px);
        for (std::size_t i = 8; i < N - r; i+=8) {
            __m256 vecx = _mm256_loadu_ps(px + i);
            maxVec = _mm256_max_ps(vecx, maxVec);
        }
        /* find max value in result */
        float result[8];
        _mm256_store_ps(result, maxVec);
        float maxValue = result[0];
        for (std::size_t i = 1; i < 8; i++) {
            maxValue = result[i] > maxValue ? result[i] : maxValue;
        }
        /* find max value in the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            maxValue = px[i] > maxValue ? px[i] : maxValue;
        }
        return maxValue;
    }


    inline static double min(const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        /* align */
        std::size_t r = N%4;
        /* find min value per 8 element */
        __m256d minVec = _mm256_loadu_pd(px);
        for (std::size_t i = 4; i < N - r; i+=4) {
            __m256d vecx = _mm256_loadu_pd(px + i);
            minVec = _mm256_min_pd(vecx, minVec);
        }
        /* find min value in result */
        double result[4];
        _mm256_store_pd(result, minVec);
        double minValue = result[0];
        for (std::size_t i = 1; i < 4; i++) {
            minValue = result[i] > minValue ? result[i] : minValue;
        }
        /* find min value in the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            minValue = px[i] > minValue ? px[i] : minValue;
        }
        return minValue;
    }

    inline static float min(const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        /* align */
        std::size_t r = N%8;
        /* find min value per 8 element */
        __m256 minVec = _mm256_loadu_ps(px);
        for (std::size_t i = 8; i < N - r; i+=8) {
            __m256 vecx = _mm256_loadu_ps(px + i);
            minVec = _mm256_min_ps(vecx, minVec);
        }
        /* find min value in result */
        float result[8];
        _mm256_store_ps(result, minVec);
        float minValue = result[0];
        for (std::size_t i = 1; i < 8; i++) {
            minValue = result[i] > minValue ? result[i] : minValue;
        }
        /* find min value in the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            minValue = px[i] > minValue ? px[i] : minValue;
        }
        return minValue;
    }

    inline static double sum(const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        /* init */
        std::size_t r = N%4;
        double result[4] = {0};
        __m256d vecs = _mm256_loadu_pd(result);
        for (std::size_t i = 0; i < N - r; i+=4) {
            __m256d vecx = _mm256_loadu_pd(px + i);
            vecs = _mm256_add_pd(vecs, vecx);
        }
        /* sum up result */
        _mm256_store_pd(result, vecs);
        double s = 0;
        for (std::size_t i = 0; i < 4; i++) {
            s += result[i];
        }
        /* sum up the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            s += px[i];
        }
        return s;
    }

    inline static float sum(const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        /* init */
        std::size_t r = N%8;
        float result[8] = {0};
        __m256 vecs = _mm256_setzero_ps();
        for (std::size_t i = 0; i < N - r; i+=8) {
            __m256 vecx = _mm256_loadu_ps(px + i);
            vecs = _mm256_add_ps(vecs, vecx);
        }
        /* sum up result */
        /* _mm256_hadd_ps: horizontal add */
        vecs = _mm256_hadd_ps(vecs, vecs);
        vecs = _mm256_hadd_ps(vecs, vecs);
        _mm256_store_ps(result, vecs);
        float s = result[0] + result[4];
        /* sum up the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            s += px[i];
        }
        return s;
    }

    inline static void sqrt(double* __restrict y, const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        double *py = y;
        std::size_t r = N%4;
        __m256d vecx;
        __m256d vecy;
        for (std::size_t i = 0; i < N - r; i+=4) {
            vecx = _mm256_loadu_pd(px + i);
            vecy = _mm256_sqrt_pd(vecx);
            _mm256_store_pd(py + i, vecy);
        }
        /* sqrt the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            py[i] += std::sqrt(px[i]);
        }
        return;
    }

    inline static void sqrt(float* __restrict y, const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        float *py = y;
        std::size_t r = N%8;
        __m256 vecx;
        __m256 vecy;
        for (std::size_t i = 0; i < N - r; i+=8) {
            vecx = _mm256_loadu_ps(px + i);
            vecy = _mm256_sqrt_ps(vecx);
            _mm256_store_ps(py + i, vecy);
        }
        /* sqrt the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            py[i] += std::sqrt(px[i]);
        }
        return;
    }

    inline static void dot(double* __restrict y,
                           const double* __restrict x1,
                           const double* __restrict x2,
                           std::size_t N)
    {
        const double *px1 = x1;
        const double *px2 = x2;
        double *py = y;
        std::size_t r = N%4;
        __m256d vecx1;
        __m256d vecx2;
        __m256d vecy;
        for (std::size_t i = 0; i < N - r; i+=4) {
            vecx1 = _mm256_loadu_pd(px1 + i);
            vecx2 = _mm256_loadu_pd(px2 + i);
            vecy = _mm256_fmadd_pd(vecx1, vecx2, vecy);
            _mm256_store_pd(py + i, vecy);
        }
        /* dot the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            py[i] += px1[i]*px2[i];
        }
        return;
    }

    inline static void dot(float* __restrict y,
                           const float* __restrict x1,
                           const float* __restrict x2,
                           std::size_t N)
    {
        const float *px1 = x1;
        const float *px2 = x2;
        float *py = y;
        std::size_t r = N%8;
        __m256 vecx1;
        __m256 vecx2;
        __m256 vecy;
        for (std::size_t i = 0; i < N - r; i+=8) {
            vecx1 = _mm256_loadu_ps(px1 + i);
            vecx2 = _mm256_loadu_ps(px2 + i);
            vecy = _mm256_fmadd_ps(vecx1, vecx2, vecy);
            _mm256_store_ps(py + i, vecy);
        }
        /* dot the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            py[i] += px1[i]*px2[i];
        }
        return;
    }

    inline static void norm2s(double* __restrict y,
                              const double* __restrict x1,
                              const double* __restrict x2,
                              std::size_t N)
    {
        const double *px1 = x1;
        const double *px2 = x2;
        double *py = y;
        std::size_t r = N%4;
        __m256d vecx1;
        __m256d vecx2;
        __m256d vecx;
        __m256d vecy;
        for (std::size_t i = 0; i < N - r; i+=4) {
            vecx1 = _mm256_loadu_pd(px1 + i);
            vecx2 = _mm256_loadu_pd(px2 + i);
            vecx = _mm256_sub_pd(vecx1, vecx2);
            vecy = _mm256_fmadd_pd(vecx, vecx, vecy);
            _mm256_store_pd(py + i, vecy);
        }
        /* the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            py[i] += (px1[i] - px2[i])*(px1[i] - px2[i]);
        }
        return;
    }

    inline static void norm2s(float* __restrict y,
                              const float* __restrict x1,
                              const float* __restrict x2,
                              std::size_t N)
    {
        const float *px1 = x1;
        const float *px2 = x2;
        float *py = y;
        std::size_t r = N%8;
        __m256 vecx1;
        __m256 vecx2;
        __m256 vecx;
        __m256 vecy;
        for (std::size_t i = 0; i < N - r; i+=8) {
            vecx1 = _mm256_loadu_ps(px1 + i);
            vecx2 = _mm256_loadu_ps(px2 + i);
            vecx = _mm256_sub_ps(vecx1, vecx2);
            vecy = _mm256_fmadd_ps(vecx, vecx, vecy);
            _mm256_store_ps(py + i, vecy);
        }
        /* the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            py[i] += (px1[i] - px2[i])*(px1[i] - px2[i]);
        }
        return;
    }

    inline static void matMul4(double* __restrict z, std::size_t zRow, std::size_t zCol,
                               const double* __restrict x, std::size_t xRow, std::size_t xCol,
                               const double* __restrict y, std::size_t yRow, std::size_t yCol)
    {
        const double *x_ = x;
        const double *y_ = y;
        double *z_ = z;
        std::size_t r = zCol%4;
        for (std::size_t i = 0; i < zRow; i++) {
            for (std::size_t k = 0; k < xCol; k++) {
                double xik = x_[i*xCol + k];
                __m256d vecx = _mm256_set1_pd(xik);
                for (std::size_t j = 0; j < zCol - r; j+=4) {
                    __m256d vecy = _mm256_loadu_pd(y_ + k*yCol + j);
                    __m256d vecz = _mm256_loadu_pd(z_ + i*xCol + j);
                    /* _mm256_fmadd_ps(a, b, c): a*b + c */
                    vecz = _mm256_fmadd_pd(vecx, vecy, vecz);
                    /* store result */
                    _mm256_store_pd(z_ + i*xCol + j, vecz);
                }
                for (std::size_t j = zCol - r; j < zCol; j++) {
                    z_[i*zCol + j] += xik * y_[k*yCol + j];
                }
            }
        }
        return;
    }

    inline static void matMul8(float* __restrict z, std::size_t zRow, std::size_t zCol,
                               const float* __restrict x, std::size_t xRow, std::size_t xCol,
                               const float* __restrict y, std::size_t yRow, std::size_t yCol)
    {
        /*
            origin:
                https://blog.csdn.net/StandCrow/article/details/120206063
        */
        const float *x_ = x;
        const float *y_ = y;
        float *z_ = z;
        std::size_t r = zCol%8;
        for (std::size_t i = 0; i < zRow; i++) {
            for (std::size_t k = 0; k < xCol; k++) {
                float xik = x_[i*xCol + k];
                __m256 vecx = _mm256_set1_ps(xik);
                for (std::size_t j = 0; j < zCol - r; j+=8) {
                    /* put 8 float into __m256 */
                    __m256 vecy = _mm256_loadu_ps(y_ + k*yCol + j);
                    __m256 vecz = _mm256_loadu_ps(z_ + i*xCol + j);
                    /* _mm256_fmadd_ps(a, b, c): a*b + c */
                    vecz = _mm256_fmadd_ps(vecx, vecy, vecz);
                    /* store result */
                    _mm256_store_ps(z_ + i*xCol + j, vecz);
                }
                for (std::size_t j = zCol - r; j < zCol; j++) {
                    z_[i*zCol + j] += xik * y_[k*yCol + j];
                }
            }
        }
        return;
    }
#endif // AVX2
};


template<std::size_t Ni, typename Instruction=SSE2,typename T=double, std::size_t align=16>
class Vector
{
public:
    using Type = T;
    constexpr static std::size_t N = Ni;
public:
    T* val;
public:
    Vector():val(nullptr)
    {
        val = static_cast<T*>(_mm_malloc(sizeof (double)*N, align));
    }
    ~Vector()
    {
        if (val != nullptr) {
            _mm_free(val);
            val = nullptr;
        }
    }

    inline std::size_t size() const {return N;}

    Vector operator + (const Vector &x) const
    {
        Vector<N> y;
        Instruction::add(y.val, val, x.val, N);
        return y;
    }

    Vector operator - (const Vector &x) const
    {
        Vector<N> y;
        Instruction::sub(y.val, val, x.val, N);
        return y;
    }

    Vector operator * (const Vector &x) const
    {
        Vector<N> y;
        Instruction::mul(y.val, val, x.val, N);
        return y;
    }

    Vector operator / (const Vector &x) const
    {
        Vector<N> y;
        Instruction::div(y.val, val, x.val, N);
        return y;
    }
};


}


#endif // SIMD_HPP
