#ifndef SSE2FUNC_HPP
#define SSE2FUNC_HPP
#include <immintrin.h>
#include <vector>
#include <cmath>

namespace simd {
struct SSE2 {

    static bool check()
    {
        return true;
    }
#if defined(__SSE2__)


    template<typename T>
    struct Unit {
        constexpr static std::size_t value = 0;
    };
    template<>
    struct Unit<double> {
        constexpr static std::size_t value = sizeof (__m128d)/sizeof (double);
    };
    template<>
    struct Unit<float> {
        constexpr static std::size_t value = sizeof (__m128)/sizeof (float);
    };

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

    inline static void matMul(double* __restrict z, std::size_t zRow, std::size_t zCol,
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

    inline static void matMul(float* __restrict z, std::size_t zRow, std::size_t zCol,
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
}
#endif // SSE2FUNC_HPP
