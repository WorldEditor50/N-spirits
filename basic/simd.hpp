#ifndef SIMD_HPP
#define SIMD_HPP
#include <immintrin.h>
#include <vector>

namespace simd {

struct SSE2 {

    inline static void add(double *z, double *y, double *x, std::size_t N)
    {
        double *px = x;
        double *py = y;
        double *pz = z;
        /* __m128d: SSE2 double */
        __m128d vecx;
        __m128d vecy;
        __m128d vecz;
        for (std::size_t i = 0; i < N/2; i++) {
            /* put 2 double into __m128d */
            vecx = _mm_load_pd(px);
            vecy = _mm_load_pd(py);
            /* add */
            vecz = _mm_add_pd(vecx, vecy);
            /* store result */
            _mm_store_pd(pz, vecz);
            /* move */
            px += 2;
            py += 2;
            pz += 2;
        }
        return;
    }

    inline static void sub(double *z, double *y, double *x, std::size_t N)
    {
        double *px = x;
        double *py = y;
        double *pz = z;
        /* __m128d: SSE2 double */
        __m128d vecx;
        __m128d vecy;
        __m128d vecz;
        for (std::size_t i = 0; i < N/2; i++) {
            /* put 2 double into __m128d */
            vecx = _mm_load_pd(px);
            vecy = _mm_load_pd(py);
            /* sub */
            vecz = _mm_sub_pd(vecx, vecy);
            /* store result */
            _mm_store_pd(pz, vecz);
            /* move */
            px += 2;
            py += 2;
            pz += 2;
        }
        return;
    }

    inline static void mul(double *z, double *y, double *x, std::size_t N)
    {
        double *px = x;
        double *py = y;
        double *pz = z;
        /* __m128d: SSE2 double */
        __m128d vecx;
        __m128d vecy;
        __m128d vecz;
        for (std::size_t i = 0; i < N/2; i++) {
            /* put 2 double into __m128d */
            vecx = _mm_load_pd(px);
            vecy = _mm_load_pd(py);
            /* multipy */
            vecz = _mm_mul_pd(vecx, vecy);
            /* store result */
            _mm_store_pd(pz, vecz);
            /* move */
            px += 2;
            py += 2;
            pz += 2;
        }
        return;
    }

    inline static void div(double *z, double *y, double *x, std::size_t N)
    {
        double *px = x;
        double *py = y;
        double *pz = z;
        /* __m128d: SSE2 double */
        __m128d vecx;
        __m128d vecy;
        __m128d vecz;
        for (std::size_t i = 0; i < N/2; i++) {
            /* put 2 double into __m128d */
            vecx = _mm_load_pd(px);
            vecy = _mm_load_pd(py);
            /* divide */
            vecz = _mm_div_pd(vecx, vecy);
            /* store result */
            _mm_store_pd(pz, vecz);
            /* move */
            px += 2;
            py += 2;
            pz += 2;
        }
        return;
    }

    inline static void add(float *z, float *y, float *x, std::size_t N)
    {
        float *px = x;
        float *py = y;
        float *pz = z;
        /* __m128: SSE2 float */
        __m128 vecx;
        __m128 vecy;
        __m128 vecz;
        for (std::size_t i = 0; i < N/2; i++) {
            /* put 2 float into __m128 */
            vecx = _mm_loadu_ps(px);
            vecy = _mm_loadu_ps(py);
            /* add */
            vecz = _mm_add_ps(vecx, vecy);
            /* store result */
            _mm_storeu_ps(pz, vecz);
            /* move */
            px += 2;
            py += 2;
            pz += 2;
        }
        return;
    }

    inline static void sub(float *z, float *y, float *x, std::size_t N)
    {
        float *px = x;
        float *py = y;
        float *pz = z;
        /* __m128: SSE2 float */
        __m128 vecx;
        __m128 vecy;
        __m128 vecz;
        for (std::size_t i = 0; i < N/2; i++) {
            /* put 2 float into __m128 */
            vecx = _mm_loadu_ps(px);
            vecy = _mm_loadu_ps(py);
            /* add */
            vecz = _mm_sub_ps(vecx, vecy);
            /* store result */
            _mm_storeu_ps(pz, vecz);
            /* move */
            px += 2;
            py += 2;
            pz += 2;
        }
        return;
    }

    inline static void mul(float *z, float *y, float *x, std::size_t N)
    {
        float *px = x;
        float *py = y;
        float *pz = z;
        /* __m128: SSE2 float */
        __m128 vecx;
        __m128 vecy;
        __m128 vecz;
        for (std::size_t i = 0; i < N/2; i++) {
            /* put 2 float into __m128 */
            vecx = _mm_loadu_ps(px);
            vecy = _mm_loadu_ps(py);
            /* mul */
            vecz = _mm_mul_ps(vecx, vecy);
            /* store result */
            _mm_storeu_ps(pz, vecz);
            /* move */
            px += 2;
            py += 2;
            pz += 2;
        }
        return;
    }

    inline static void div(float *z, float *y, float *x, std::size_t N)
    {
        float *px = x;
        float *py = y;
        float *pz = z;
        /* __m128: SSE2 float */
        __m128 vecx;
        __m128 vecy;
        __m128 vecz;
        for (std::size_t i = 0; i < N/2; i++) {
            /* put 2 float into __m128 */
            vecx = _mm_loadu_ps(px);
            vecy = _mm_loadu_ps(py);
            /* div */
            vecz = _mm_div_ps(vecx, vecy);
            /* store result */
            _mm_storeu_ps(pz, vecz);
            /* move */
            px += 2;
            py += 2;
            pz += 2;
        }
        return;
    }

};

/*
   avx2 on windows:
        implement command "bcdedit/set xsavedisable 0" and set compiler option: "/arch:AVX2"
        https://devblogs.microsoft.com/cppblog/avx2-support-in-visual-studio-c-compiler/

   avx2 on linux: sudo apt-get install libmkl-dev libmkl-avx
*/

struct AVX2 {
    inline static void add(double *z, double *y, double *x, std::size_t N)
    {
        double *px = x;
        double *py = y;
        double *pz = z;
        /* __m256d: AVX double */
        __m256d vecx;
        __m256d vecy;
        __m256d vecz;
        for (std::size_t i = 0; i < N/4; i++) {
            /* put 4 double into __m256d */
            vecx = _mm256_load_pd(px);
            vecy = _mm256_load_pd(py);
            /* add */
            vecz = _mm256_add_pd(vecx, vecy);
            /* store result */
            _mm256_store_pd(pz, vecz);
            /* move */
            px += 4;
            py += 4;
            pz += 4;
        }
        return;
    }

    inline static void sub(double *z, double *y, double *x, std::size_t N)
    {
        double *px = x;
        double *py = y;
        double *pz = z;
        /* __m256d: AVX double */
        __m256d vecx;
        __m256d vecy;
        __m256d vecz;
        for (std::size_t i = 0; i < N/4; i++) {
            /* put 4 double into __m256d */
            vecx = _mm256_load_pd(px);
            vecy = _mm256_load_pd(py);
            /* add */
            vecz = _mm256_sub_pd(vecx, vecy);
            /* store result */
            _mm256_store_pd(pz, vecz);
            /* move */
            px += 4;
            py += 4;
            pz += 4;
        }
        return;
    }

    inline static void mul(double *z, double *y, double *x, std::size_t N)
    {
        double *px = x;
        double *py = y;
        double *pz = z;
        /* __m256d: AVX double */
        __m256d vecx;
        __m256d vecy;
        __m256d vecz;
        for (std::size_t i = 0; i < N/4; i++) {
            /* put 4 double into __m256d */
            vecx = _mm256_load_pd(px);
            vecy = _mm256_load_pd(py);
            /* mul */
            vecz = _mm256_mul_pd(vecx, vecy);
            /* store result */
            _mm256_store_pd(pz, vecz);
            /* move */
            px += 4;
            py += 4;
            pz += 4;
        }
        return;
    }

    inline static void div(double *z, double *y, double *x, std::size_t N)
    {
        double *px = x;
        double *py = y;
        double *pz = z;
        /* __m256d: AVX double */
        __m256d vecx;
        __m256d vecy;
        __m256d vecz;
        for (std::size_t i = 0; i < N/4; i++) {
            /* put 4 double into __m256d */
            vecx = _mm256_load_pd(px);
            vecy = _mm256_load_pd(py);
            /* div */
            vecz = _mm256_div_pd(vecx, vecy);
            /* store result */
            _mm256_store_pd(pz, vecz);
            /* move */
            px += 4;
            py += 4;
            pz += 4;
        }
        return;
    }

    inline static void add(float *z, float *y, float *x, std::size_t N)
    {
        float *px = x;
        float *py = y;
        float *pz = z;
        /* __m256: AVX2 float */
        __m256 vecx;
        __m256 vecy;
        __m256 vecz;
        for (std::size_t i = 0; i < N/4; i++) {
            /* put 4 float into __m256 */
            vecx = _mm256_loadu_ps(px);
            vecy = _mm256_loadu_ps(py);
            /* add */
            vecz = _mm256_add_ps(vecx, vecy);
            /* store result */
            _mm256_storeu_ps(pz, vecz);
            /* move */
            px += 4;
            py += 4;
            pz += 4;
        }
    }

    inline static void sub(float *z, float *y, float *x, std::size_t N)
    {
        float *px = x;
        float *py = y;
        float *pz = z;
        /* __m256: AVX2 float */
        __m256 vecx;
        __m256 vecy;
        __m256 vecz;
        for (std::size_t i = 0; i < N/4; i++) {
            /* put 4 float into __m256 */
            vecx = _mm256_loadu_ps(px);
            vecy = _mm256_loadu_ps(py);
            /* add */
            vecz = _mm256_sub_ps(vecx, vecy);
            /* store result */
            _mm256_storeu_ps(pz, vecz);
            /* move */
            px += 4;
            py += 4;
            pz += 4;
        }
    }

    inline static void mul(float *z, float *y, float *x, std::size_t N)
    {
        float *px = x;
        float *py = y;
        float *pz = z;
        /* __m256: AVX2 float */
        __m256 vecx;
        __m256 vecy;
        __m256 vecz;
        for (std::size_t i = 0; i < N/4; i++) {
            /* put 4 float into __m256 */
            vecx = _mm256_loadu_ps(px);
            vecy = _mm256_loadu_ps(py);
            /* mul */
            vecz = _mm256_mul_ps(vecx, vecy);
            /* store result */
            _mm256_storeu_ps(pz, vecz);
            /* move */
            px += 4;
            py += 4;
            pz += 4;
        }
    }

    inline static void div(float *z, float *y, float *x, std::size_t N)
    {
        float *px = x;
        float *py = y;
        float *pz = z;
        /* __m256: AVX2 float */
        __m256 vecx;
        __m256 vecy;
        __m256 vecz;
        for (std::size_t i = 0; i < N/4; i++) {
            /* put 4 float into __m256 */
            vecx = _mm256_loadu_ps(px);
            vecy = _mm256_loadu_ps(py);
            /* div */
            vecz = _mm256_div_ps(vecx, vecy);
            /* store result */
            _mm256_storeu_ps(pz, vecz);
            /* move */
            px += 4;
            py += 4;
            pz += 4;
        }
        return;
    }

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
