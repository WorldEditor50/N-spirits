#ifndef AVX2WRAPPER_HPP
#define AVX2WRAPPER_HPP

#include <immintrin.h>
#include <vector>
#include <cmath>

/*
    Start:
        avx2 on windows:
            1. implement command "bcdedit/set xsavedisable 0" and set compiler option: "/arch:AVX2"
            2. https://devblogs.microsoft.com/cppblog/avx2-support-in-visual-studio-c-compiler/

        avx2 on linux:
            sudo apt-get install libmkl-dev libmkl-avx

    Note:
       1. when compile with msvc debug mode, compiler may not perform inlining ,
          the wrapper will cost double time
       2. memory alignment is easy to make simd running in parallel
       3. use reference or pointer of variable will make function run faster
       4. use keyword __restrict、 noexcept
       5. simd operations:
            a. load or set values
            b. process values
            b. store values

    Warning:
        1. this wrapper will work while the data size is greater than

    Resource:
        1. https://www.cs.virginia.edu/~cr4bd/3330/F2018/simdref.html
*/

#ifdef _MSC_VER
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE __attribute__((always_inline)) inline
#endif

namespace wrap {

template<typename T>
class AVX2
{
public:
    using Type = T;

#if defined(__AVX2__)


    struct M256d {

        FORCE_INLINE static __m256d load(double const* __restrict __p) noexcept
        {
            return _mm256_load_pd(__p);
        }
        FORCE_INLINE static __m256d loadu(double const* __restrict __p) noexcept
        {
            return _mm256_loadu_pd(__p);
        }
        FORCE_INLINE static __m256d set1(double& __restrict __w) noexcept
        {
            return _mm256_set1_pd(__w);
        }
        FORCE_INLINE static __m256d zero() noexcept
        {
            return _mm256_setzero_pd();
        }
        FORCE_INLINE static void store(double* __restrict __p, __m256d& __restrict __a) noexcept
        {
            return _mm256_store_pd(__p, __a);
        }
        FORCE_INLINE static void storeu(double* __restrict __p, __m256d& __restrict __a) noexcept
        {
            return _mm256_storeu_pd(__p, __a);
        }
        FORCE_INLINE static __m256d add(__m256d& __restrict __a, __m256d& __restrict __b) noexcept
        {
            return _mm256_add_pd(__a, __b);
        }
        FORCE_INLINE static __m256d sub(__m256d& __restrict __a, __m256d& __restrict __b) noexcept
        {
            return _mm256_sub_pd(__a, __b);
        }
        FORCE_INLINE static __m256d mul(__m256d& __restrict __a, __m256d& __restrict __b) noexcept
        {
            return _mm256_mul_pd(__a, __b);
        }
        FORCE_INLINE static __m256d div(__m256d& __restrict __a, __m256d& __restrict __b) noexcept
        {
            return _mm256_div_pd(__a, __b);
        }
        FORCE_INLINE static __m256d hadd(__m256d& __restrict __a, __m256d& __restrict __b) noexcept
        {
            return _mm256_hadd_pd(__a, __b);
        }
        FORCE_INLINE static __m256d hsub(__m256d& __restrict __a, __m256d& __restrict __b) noexcept
        {
            return _mm256_hsub_pd(__a, __b);
        }
        FORCE_INLINE static __m256d fmadd(__m256d& __restrict __a, __m256d& __restrict __b, __m256& __restrict __c) noexcept
        {
            return _mm256_fmadd_pd(__a, __b, __c);
        }
        FORCE_INLINE static __m256d fmsub(__m256d& __restrict __a, __m256d& __restrict __b, __m256& __restrict __c) noexcept
        {
            return _mm256_fmsub_pd(__a, __b, __c);
        }

        FORCE_INLINE static __m256d max(__m256d& __restrict __a, __m256d& __restrict __b) noexcept
        {
            return _mm256_max_pd(__a, __b);
        }
        FORCE_INLINE static __m256d min(__m256d& __restrict __a, __m256d& __restrict __b) noexcept
        {
            return _mm256_min_pd(__a, __b);
        }
        FORCE_INLINE static __m256d sqrt(__m256d& __restrict __a) noexcept
        {
            return _mm256_sqrt_pd(__a);
        }

        /* rewrap */
        using Add = __m256d (*)(__m256d& __restrict , __m256d& __restrict );
        using Sub = __m256d (*)(__m256d& __restrict , __m256d& __restrict );
        using Mul = __m256d (*)(__m256d& __restrict , __m256d& __restrict );
        using Div = __m256d (*)(__m256d& __restrict , __m256d& __restrict );
        using Max = __m256d (*)(__m256d& __restrict , __m256d& __restrict );
        using Min = __m256d (*)(__m256d& __restrict , __m256d& __restrict );

    };

    struct M256 {
        /* wrapper */
#if 0
        FORCE_INLINE static void load(__m256& __restrict __r, float const* __restrict __p) noexcept
        {
            __r = _mm256_load_ps(__p);
            return;
        }
        FORCE_INLINE static void loadu(__m256& __restrict __r, float const* __restrict __p) noexcept
        {
            __r = _mm256_loadu_ps(__p);
        }
        FORCE_INLINE static void set1(__m256& __restrict __r, float& __restrict __w) noexcept
        {
            __r = _mm256_set1_ps(__w);
        }
        FORCE_INLINE static void zero(__m256& __restrict __r) noexcept
        {
            __r = _mm256_setzero_ps();
        }
        FORCE_INLINE static void store(float* __restrict __p, __m256& __restrict __a) noexcept
        {
            return _mm256_store_ps(__p, __a);
        }
        FORCE_INLINE static void storeu(float* __restrict __p, __m256& __restrict __a) noexcept
        {
            return _mm256_storeu_ps(__p, __a);
        }
        FORCE_INLINE static void add(__m256& __restrict __r, __m256& __restrict __a, __m256& __restrict __b) noexcept
        {
            __r = _mm256_add_ps(__a, __b);
        }
        FORCE_INLINE static void sub(__m256& __restrict __r, __m256& __restrict __a, __m256& __restrict __b) noexcept
        {
            __r = _mm256_sub_ps(__a, __b);
        }
        FORCE_INLINE static void mul(__m256& __restrict __r,  __m256& __restrict __a, __m256& __restrict __b) noexcept
        {
            __r = _mm256_mul_ps(__a, __b);
        }
        FORCE_INLINE static void div(__m256& __restrict __r, __m256& __restrict __a, __m256& __restrict __b) noexcept
        {
            __r = _mm256_div_ps(__a, __b);
        }
        FORCE_INLINE static void hadd(__m256& __restrict __r, __m256& __restrict __a, __m256& __restrict __b) noexcept
        {
            __r = _mm256_hadd_ps(__a, __b);
        }
        FORCE_INLINE static void hsub(__m256& __restrict __r, __m256& __restrict __a, __m256& __restrict __b) noexcept
        {
            __r = _mm256_hsub_ps(__a, __b);
        }
        FORCE_INLINE static void fmadd(__m256& __restrict __r, __m256& __restrict __a, __m256& __restrict __b, __m256& __restrict __c) noexcept
        {
            __r = _mm256_fmadd_ps(__a, __b, __c);
        }
        FORCE_INLINE static void fmsub(__m256& __restrict __r, __m256& __restrict __a, __m256& __restrict __b, __m256& __restrict __c) noexcept
        {
            __r = _mm256_fmsub_ps(__a, __b, __c);
        }

        FORCE_INLINE static void max(__m256& __restrict __r, __m256& __restrict __a, __m256& __restrict __b) noexcept
        {
            __r = _mm256_max_ps(__a, __b);
        }
        FORCE_INLINE static void min(__m256& __restrict __r, __m256& __restrict __a, __m256& __restrict __b) noexcept
        {
            __r = _mm256_min_ps(__a, __b);
        }
        FORCE_INLINE static void sqrt(__m256& __restrict __r, __m256& __restrict __a) noexcept
        {
            __r = _mm256_sqrt_ps(__a);
            return;
        }
#endif

        FORCE_INLINE static __m256 load(float const* __restrict __p) noexcept
        {
            return _mm256_load_ps(__p);
        }
        FORCE_INLINE static __m256 loadu(float const* __restrict __p) noexcept
        {
            return _mm256_loadu_ps(__p);
        }
        FORCE_INLINE static __m256 set1(float& __restrict __w) noexcept
        {
            return _mm256_set1_ps(__w);
        }
        FORCE_INLINE static __m256 zero() noexcept
        {
            return _mm256_setzero_ps();
        }
        FORCE_INLINE static void store(float* __restrict __p, __m256& __restrict __a) noexcept
        {
            return _mm256_store_ps(__p, __a);
        }
        FORCE_INLINE static void storeu(float* __restrict __p, __m256& __restrict __a) noexcept
        {
            return _mm256_storeu_ps(__p, __a);
        }
        FORCE_INLINE static __m256 add(__m256& __restrict __a, __m256& __restrict __b) noexcept
        {
            return _mm256_add_ps(__a, __b);
        }
        FORCE_INLINE static __m256 sub(__m256& __restrict __a, __m256& __restrict __b) noexcept
        {
            return _mm256_sub_ps(__a, __b);
        }
        FORCE_INLINE static __m256 mul(__m256& __restrict __a, __m256& __restrict __b) noexcept
        {
            return _mm256_mul_ps(__a, __b);
        }
        FORCE_INLINE static __m256 div(__m256& __restrict __a, __m256& __restrict __b) noexcept
        {
            return _mm256_div_ps(__a, __b);
        }
        FORCE_INLINE static __m256 hadd(__m256& __restrict __a, __m256& __restrict __b) noexcept
        {
            return _mm256_hadd_ps(__a, __b);
        }
        FORCE_INLINE static __m256 hsub( __m256& __restrict __a, __m256& __restrict __b) noexcept
        {
            return _mm256_hsub_ps(__a, __b);
        }
        FORCE_INLINE static __m256 fmadd(__m256& __restrict __a, __m256& __restrict __b, __m256& __restrict __c) noexcept
        {
            return _mm256_fmadd_ps(__a, __b, __c);
        }
        FORCE_INLINE static __m256 fmsub(__m256& __restrict __a, __m256& __restrict __b, __m256& __restrict __c) noexcept
        {
            return _mm256_fmsub_ps(__a, __b, __c);
        }

        FORCE_INLINE static __m256 max(__m256& __restrict __a, __m256& __restrict __b) noexcept
        {
            return _mm256_max_ps(__a, __b);
        }
        FORCE_INLINE static __m256 min(__m256& __restrict __a, __m256& __restrict __b) noexcept
        {
            return _mm256_min_ps(__a, __b);
        }
        FORCE_INLINE static __m256 sqrt(__m256& __restrict __a) noexcept
        {
            return _mm256_sqrt_ps(__a);
        }
        /* rewrap */
        using Add = __m256 (*)(__m256& __restrict , __m256& __restrict );
        using Sub = __m256 (*)(__m256& __restrict , __m256& __restrict );
        using Mul = __m256 (*)(__m256& __restrict , __m256& __restrict );
        using Div = __m256 (*)(__m256& __restrict , __m256& __restrict );
        using Max = __m256 (*)(__m256& __restrict , __m256& __restrict );
        using Min = __m256 (*)(__m256& __restrict , __m256& __restrict );

    };

    template<typename Ti>
    struct Selector {};
    template<>
    struct Selector<double> {
        using Type = __m256d;
        using Func = M256d;
        constexpr static std::size_t step = sizeof (__m256)/sizeof (float);
    };
    template<>
    struct Selector<float> {
        using Type = __m256;
        using Func = M256;
        constexpr static std::size_t step = sizeof (__m256)/sizeof (float);
    };
    template<>
    struct Selector<int> {
        using Type = __m256i;
        constexpr static std::size_t step = sizeof (__m256i)/sizeof (int);
    };

    /* select */
    using m256 = typename Selector<T>::Type;
    constexpr static std::size_t step = Selector<T>::step;
    using M256Func = typename Selector<T>::Func;

    /* align basic operator */
    struct AddOp {
        constexpr static typename M256Func::Add eval = M256Func::add;
        inline static T eval_(T x1, T x2) {return x1 + x2;};
    };
    struct SubOp {
        constexpr static typename M256Func::Sub eval = M256Func::sub;
        inline static T eval_(T x1, T x2) {return x1 - x2;};
    };
    struct MulOp {
        constexpr static typename M256Func::Mul eval = M256Func::mul;
        inline static T eval_(T x1, T x2) {return x1 * x2;};
    };
    struct DivOp {
        constexpr static typename M256Func::Div eval = M256Func::div;
        inline static T eval_(T x1, T x2) {return x1 / x2;};
    };

    template<typename Op>
    struct Evaluate {
        inline static void impl(T* __restrict z, const T* __restrict y, const T* __restrict x, std::size_t N)
        {
            const T *px = x;
            const T *py = y;
            T *pz = z;
            m256 vecx;
            m256 vecy;
            m256 vecz;
            std::size_t r = N%step;
            for (std::size_t i = 0; i < N - r; i+= step) {
                vecx = M256Func::load(px + i);
                vecy = M256Func::load(py + i);
                vecz = Op::eval(vecx, vecy);
                M256Func::storeu(pz + i, vecz);
            }
            /* the rest of element */
            for (std::size_t i = N - r; i < N; i++) {
                z[i] = Op::eval_(y[i], x[i]);
            }
            return;
        }

        inline static void impl(T* __restrict z, const T* __restrict y, T x, std::size_t N)
        {
            const T *py = y;
            float *pz = z;
            /* __m256: AVX2 float */
            m256 vecx = M256Func::set1(x);
            m256 vecy;
            m256 vecz;
            std::size_t r = N%step;
            for (std::size_t i = 0; i < N - r; i+= step) {
                /* put 8 float into __m256 */
                vecy = M256Func::load(py + i);
                /* op */
                vecz = Op::eval(vecy, vecx);
                /* store result */
                M256Func::storeu(pz + i, vecz);
            }
            /* the rest of element */
            for (std::size_t i = N - r; i < N; i++) {
                z[i] = Op::eval_(y[i], x);
            }
            return;
        }

    };

    inline static void add(T* __restrict z, const T* __restrict y, const T* __restrict x, std::size_t N)
    {
        return Evaluate<AddOp>::impl(z, y, x, N);
    }

    inline static void add(T* __restrict z, const T* __restrict y,  T x, std::size_t N)
    {
        return Evaluate<AddOp>::impl(z, y, x, N);
    }

    inline static void sub(T* __restrict z, const T* __restrict y, const T* __restrict x, std::size_t N)
    {
        return Evaluate<SubOp>::impl(z, y, x, N);
    }

    inline static void sub(T* __restrict z, const T* __restrict y,  T x, std::size_t N)
    {
        return Evaluate<SubOp>::impl(z, y, x, N);
    }

    inline static void mul(T* __restrict z, const T* __restrict y, const T* __restrict x, std::size_t N)
    {
        return Evaluate<MulOp>::impl(z, y, x, N);
    }

    inline static void mul(T* __restrict z, const T* __restrict y,  T x, std::size_t N)
    {
        return Evaluate<MulOp>::impl(z, y, x, N);
    }

    inline static void div(T* __restrict z, const T* __restrict y, const T* __restrict x, std::size_t N)
    {
        return Evaluate<DivOp>::impl(z, y, x, N);
    }

    inline static void div(T* __restrict z, const T* __restrict y,  T x, std::size_t N)
    {
        return Evaluate<DivOp>::impl(z, y, x, N);
    }

    /* max-min */
    struct MaxOp {
        constexpr static typename M256Func::Max impl = M256Func::max;
        inline static T impl_(T x1, T x2) {return x1 > x2 ? x1 : x2;};
    };
    struct MinOp {
        constexpr static typename M256Func::Min impl = M256Func::min;
        inline static T impl_(T x1, T x2) {return x1 < x2 ? x1 : x2;};
    };

    template<typename Op>
    struct Find {
        inline static T impl(const T* __restrict x, std::size_t N)
        {
            const T *px = x;
            std::size_t r = N%step;
            /* find max value per 8 element */
            m256 vec = M256Func::loadu(px);
            for (std::size_t i = step; i < N - r; i+=step) {
                m256 vecx = M256Func::loadu(px + i);
                vec = Op::impl(vecx, vec);
            }
            /* find max value in result */
            T result[step];
            M256Func::storeu(result, vec);
            T value = result[0];
            for (std::size_t i = 1; i < step; i++) {
                value = Op::impl_(result[i], value);
            }
            /* find max value in the rest elements */
            for (std::size_t i = N - r; i < N; i++) {
                value = Op::impl_(px[i], value);
            }
            return value;
        }
    };
    inline static T max(const T* __restrict x, std::size_t N)
    {
        return Find<MaxOp>::impl(x, N);
    }

    inline static T min(const T* __restrict x, std::size_t N)
    {
        return Find<MinOp>::impl(x, N);
    }

    /* horizontal sum */
    inline static float reduce(__m256& ymm)
    {
        float result[8];
        ymm = _mm256_hadd_ps(ymm, ymm);
        ymm = _mm256_hadd_ps(ymm, ymm);
        _mm256_storeu_ps(result, ymm);
        return result[0] + result[4];
    }

    inline static double reduce(__m256d& ymm)
    {
        double result[4];
        ymm = _mm256_hadd_pd(ymm, ymm);
        _mm256_storeu_pd(result, ymm);
        return result[0] + result[2];
    }

    /* fill */
    inline static void fill(T* __restrict x, T x0, std::size_t N)
    {
        T* px = x;
        std::size_t r = N%step;
        T arrayX0[step] = {x0};
        m256 vecx0 = M256Func::loadu(arrayX0);
        for (std::size_t i = 0; i < N - r; i+=step) {
            M256Func::storeu(px + i, vecx0);
        }
        for (std::size_t i = N - r; i < N; i++) {
            x[i] = x0;
        }
        return;
    }

    /* sum */
    inline static T sum(const T* __restrict x, std::size_t N)
    {
        const T *px = x;
        /* init */
        std::size_t r = N%step;
        m256 vecs = M256Func::zero();
        for (std::size_t i = 0; i < N - r; i+=step) {
            m256 vecx = M256Func::loadu(px + i);
            vecs = M256Func::add(vecs, vecx);
        }
        /* sum up result */
        T s = reduce(vecs);
        /* sum up the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            s += px[i];
        }
        return s;
    }

    inline static void sqrt(T* __restrict y, const T* __restrict x, std::size_t N)
    {
        const T *px = x;
        T *py = y;
        std::size_t r = N%step;
        m256 vecx;
        m256 vecy;
        for (std::size_t i = 0; i < N - r; i+=step) {
            vecx = M256Func::loadu(px + i);
            vecy = M256Func::sqrt(vecx);
            M256Func::storeu(py + i, vecy);
        }
        /* the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            py[i] += std::sqrt(px[i]);
        }
        return;
    }

    inline static T dot(const T* __restrict x1, const T* __restrict x2, std::size_t N)
    {
        const T *px1 = x1;
        const T *px2 = x2;
        std::size_t r = N%step;
        m256 vecx1;
        m256 vecx2;
        m256 vecy = M256Func::zero();
        for (std::size_t i = 0; i < N - r; i+=step) {
            vecx1 = M256Func::loadu(px1 + i);
            vecx2 = M256Func::loadu(px2 + i);
            vecy = M256Func::fmadd(vecx1, vecx2, vecy);
        }
        /* sum up result */
        T s = reduce(vecy);
        /* dot the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            s += px1[i]*px2[i];
        }
        return s;
    }

    inline static T norm2s(const T* __restrict x1, const T* __restrict x2, std::size_t N)
    {
        const T *px1 = x1;
        const T *px2 = x2;
        std::size_t r = N%step;
        m256 vecx1;
        m256 vecx2;
        m256 vecx;
        m256 vecy = M256Func::zero();
        for (std::size_t i = 0; i < N - r; i+=step) {
            vecx1 = M256Func::loadu(px1 + i);
            vecx2 = M256Func::loadu(px2 + i);
            vecx = M256Func::sub(vecx1, vecx2);
            vecy = M256Func::fmadd(vecx, vecx, vecy);
        }
        /* sum up result */
        T s = reduce(vecy);
        /* the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            s += (px1[i] - px2[i])*(px1[i] - px2[i]);
        }
        return s;
    }

    inline static T variance(const T* __restrict x,  T  u, std::size_t N)
    {
        const T *px = x;
        std::size_t r = N%step;
        m256 vecx1;
        m256 vecu = M256Func::loadu(u);
        m256 vecx = M256Func::zero();
        m256 vecy = M256Func::zero();
        for (std::size_t i = 0; i < N - r; i+=step) {
            vecx1 = M256Func::loadu(px + i);
            vecx = M256Func::sub(vecx1, vecu);
            vecy = M256Func::fmadd(vecx, vecx, vecy);
        }
        /* sum up result */
        T s = reduce(vecy);
        /* the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            s += (px[i] - u)*(px[i] - u);
        }
        return s/float(N);
    }

    inline static void matMul(T* __restrict z, std::size_t zRow, std::size_t zCol,
                              const T* __restrict x, std::size_t xRow, std::size_t xCol,
                              const T* __restrict y, std::size_t yRow, std::size_t yCol)
    {
        const T *x_ = x;
        const T *y_ = y;
        T *z_ = z;
        m256 vecx;
        m256 vecy;
        m256 vecz;
        std::size_t r = zCol%step;
        for (std::size_t i = 0; i < zRow; i++) {
            for (std::size_t k = 0; k < xCol; k++) {
                T xik = x_[i*xCol + k];
                vecx = M256Func::set1(xik);
                for (std::size_t j = 0; j < zCol - r; j+=step) {
                    /* put float into m256 */
                    vecy = M256Func::loadu(y_ + k*yCol + j);
                    vecz = M256Func::loadu(z_ + i*xCol + j);
                    /* _mm256_fmadd_ps(a, b, c): a*b + c */
                    vecz = M256Func::fmadd(vecx, vecy, vecz);
                    /* store result */
                    M256Func::storeu(z_ + i*xCol + j, vecz);
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

}
#endif // AVX2WRAPPER_HPP
