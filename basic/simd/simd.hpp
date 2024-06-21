#ifndef SIMD_HPP
#define SIMD_HPP
#include <immintrin.h>
#include <vector>
#include <cmath>
#include "avx2func.hpp"
#include "sse2wrapper.hpp"
#include "avx2wrapper.hpp"

namespace simd {


template<typename T, typename I>
struct wrap {

    template<typename Ti, typename Ii>
    struct Vector_ {};
#if defined(__SSE2__)
    template<typename Ii>
    struct Vector_<double, Ii> {
        using Func = typename Ii::M128d;
        using Type = typename Func::Type;
        constexpr static std::size_t N = sizeof (Type)/sizeof (double);
    };
    template<typename Ii>
    struct Vector_<float, Ii> {
        using Func = typename Ii::M128;
        using Type = typename Func::Type;
        constexpr static std::size_t N = sizeof (Type)/sizeof (float);
    };
    template<typename Ii>
    struct Vector_<int, Ii> {

    };
#endif

#if defined(__AVX2__)
    template<typename Ii>
    struct Vector_<double, Ii> {
        using Func = typename Ii::M256d;
        using Type = typename Func::Type;
        constexpr static std::size_t N = sizeof (Type)/sizeof (double);
    };
    template<typename Ii>
    struct Vector_<float, Ii> {
        using Func = typename Ii::M256;
        using Type = typename Func::Type;
        constexpr static std::size_t N = sizeof (Type)/sizeof (float);
    };
    template<typename Ii>
    struct Vector_<int, Ii> {

    };
#endif

    /* select */
    using Vector = Vector_<T, I>;
    using VectorType = typename Vector::Type;
    /* align basic operator */
    struct AddOp {
        constexpr static typename Vector::Func::Add eval = Vector::Func::add;
        inline static T eval_(T x1, T x2) {return x1 + x2;};
    };
    struct SubOp {
        constexpr static typename Vector::Func::Sub eval = Vector::Func::sub;
        inline static T eval_(T x1, T x2) {return x1 - x2;};
    };
    struct MulOp {
        constexpr static typename Vector::Func::Mul eval = Vector::Func::mul;
        inline static T eval_(T x1, T x2) {return x1 * x2;};
    };
    struct DivOp {
        constexpr static typename Vector::Func::Div eval = Vector::Func::div;
        inline static T eval_(T x1, T x2) {return x1 / x2;};
    };

    template<typename Op>
    struct Evaluate {
        inline static void impl(T* __restrict z, const T* __restrict y, const T* __restrict x, std::size_t N)
        {
            const T *px = x;
            const T *py = y;
            T *pz = z;
            VectorType vecx;
            VectorType vecy;
            VectorType vecz;
            std::size_t r = N%Vector::N;
            for (std::size_t i = 0; i < N - r; i+= Vector::N) {
                vecx = Vector::Func::load(px + i);
                vecy = Vector::Func::load(py + i);
                vecz = Op::eval(vecx, vecy);
                Vector::Func::storeu(pz + i, vecz);
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
            VectorType vecx = Vector::Func::set1(x);
            VectorType vecy;
            VectorType vecz;
            std::size_t r = N%Vector::N;
            for (std::size_t i = 0; i < N - r; i+= Vector::N) {
                /* put 8 float into __m256 */
                vecy = Vector::Func::load(py + i);
                /* op */
                vecz = Op::eval(vecy, vecx);
                /* store result */
                Vector::Func::storeu(pz + i, vecz);
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
        constexpr static typename Vector::Func::Max impl = Vector::Func::max;
        inline static T impl_(T x1, T x2) {return x1 > x2 ? x1 : x2;};
    };
    struct MinOp {
        constexpr static typename Vector::Func::Min impl = Vector::Func::min;
        inline static T impl_(T x1, T x2) {return x1 < x2 ? x1 : x2;};
    };

    template<typename Op>
    struct Find {
        inline static T impl(const T* __restrict x, std::size_t N)
        {
            const T *px = x;
            std::size_t r = N%Vector::N;
            /* find max value per 8 element */
            VectorType vec = Vector::Func::loadu(px);
            for (std::size_t i = Vector::N; i < N - r; i+=Vector::N) {
                VectorType vecx = Vector::Func::loadu(px + i);
                vec = Op::impl(vecx, vec);
            }
            /* find max value in result */
            T result[Vector::N];
            Vector::Func::storeu(result, vec);
            T value = result[0];
            for (std::size_t i = 1; i < Vector::N; i++) {
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
        std::size_t r = N%Vector::N;
        T arrayX0[Vector::N] = {x0};
        VectorType vecx0 = Vector::Func::loadu(arrayX0);
        for (std::size_t i = 0; i < N - r; i+=Vector::N) {
            Vector::Func::storeu(px + i, vecx0);
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
        std::size_t r = N%Vector::N;
        VectorType vecs = Vector::Func::zero();
        for (std::size_t i = 0; i < N - r; i+=Vector::N) {
            VectorType vecx = Vector::Func::loadu(px + i);
            vecs = Vector::Func::add(vecs, vecx);
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
        std::size_t r = N%Vector::N;
        VectorType vecx;
        VectorType vecy;
        for (std::size_t i = 0; i < N - r; i+=Vector::N) {
            vecx = Vector::Func::loadu(px + i);
            vecy = Vector::Func::sqrt(vecx);
            Vector::Func::storeu(py + i, vecy);
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
        std::size_t r = N%Vector::N;
        VectorType vecx1;
        VectorType vecx2;
        VectorType vecy = Vector::Func::zero();
        for (std::size_t i = 0; i < N - r; i+=Vector::N) {
            vecx1 = Vector::Func::loadu(px1 + i);
            vecx2 = Vector::Func::loadu(px2 + i);
            Vector::Func::fmadd(vecx1, vecx2, vecy);
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
        std::size_t r = N%Vector::N;
        VectorType vecx1;
        VectorType vecx2;
        VectorType vecx;
        VectorType vecy = Vector::Func::zero();
        for (std::size_t i = 0; i < N - r; i+=Vector::N) {
            vecx1 = Vector::Func::loadu(px1 + i);
            vecx2 = Vector::Func::loadu(px2 + i);
            vecx = Vector::Func::sub(vecx1, vecx2);
            vecy = Vector::Func::fmadd(vecx, vecx, vecy);
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
        std::size_t r = N%Vector::N;
        VectorType vecx1;
        VectorType vecu = Vector::Func::loadu(u);
        VectorType vecx = Vector::Func::zero();
        VectorType vecy = Vector::Func::zero();
        for (std::size_t i = 0; i < N - r; i+=Vector::N) {
            vecx1 = Vector::Func::loadu(px + i);
            vecx = Vector::Func::sub(vecx1, vecu);
            Vector::Func::fmadd(vecx, vecx, vecy);
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
        VectorType vecx;
        VectorType vecy;
        VectorType vecz;
        std::size_t r = zCol%Vector::N;
        for (std::size_t i = 0; i < zRow; i++) {
            for (std::size_t k = 0; k < xCol; k++) {
                T xik = x_[i*xCol + k];
                vecx = Vector::Func::set1(xik);
                for (std::size_t j = 0; j < zCol - r; j+=Vector::N) {
                    /* put float into m256 */
                    vecy = Vector::Func::loadu(y_ + k*yCol + j);
                    vecz = Vector::Func::loadu(z_ + i*xCol + j);
                    /* _mm256_fmadd_ps(a, b, c): a*b + c */
                    Vector::Func::fmadd(vecx, vecy, vecz);
                    /* store result */
                    Vector::Func::storeu(z_ + i*xCol + j, vecz);
                }
                for (std::size_t j = zCol - r; j < zCol; j++) {
                    z_[i*zCol + j] += xik * y_[k*yCol + j];
                }
            }
        }
        return;
    }

    struct MatMul {
        inline static void ikkj(T* __restrict z, std::size_t zRow, std::size_t zCol,
                                const T* __restrict x, std::size_t xRow, std::size_t xCol,
                                const T* __restrict y, std::size_t yRow, std::size_t yCol)
        {
            const T *x_ = x;
            const T *y_ = y;
            T *z_ = z;
            VectorType vecx;
            VectorType vecy;
            VectorType vecz;
            std::size_t r = zCol%Vector::N;
            for (std::size_t i = 0; i < zRow; i++) {
                for (std::size_t k = 0; k < xCol; k++) {
                    T xik = x_[i*xCol + k];
                    vecx = Vector::Func::set1(xik);
                    for (std::size_t j = 0; j < zCol - r; j+=Vector::N) {
                        /* put float into m256 */
                        vecy = Vector::Func::loadu(y_ + k*yCol + j);
                        vecz = Vector::Func::loadu(z_ + i*xCol + j);
                        /* _mm256_fmadd_ps(a, b, c): a*b + c */
                        Vector::Func::fmadd(vecx, vecy, vecz);
                        /* store result */
                        Vector::Func::storeu(z_ + i*xCol + j, vecz);
                    }
                    for (std::size_t j = zCol - r; j < zCol; j++) {
                        z_[i*zCol + j] += xik * y_[k*yCol + j];
                    }
                }
            }
            return;
        }

        inline static void ikjk(T* __restrict z, std::size_t zRow, std::size_t zCol,
                                const T* __restrict x, std::size_t xRow, std::size_t xCol,
                                const T* __restrict y, std::size_t yRow, std::size_t yCol)
        {
            const T *x_ = x;
            const T *y_ = y;
            T *z_ = z;
            VectorType vecx;
            VectorType vecy;
            VectorType vecz;
            std::size_t r = zCol%Vector::N;
            for (std::size_t i = 0; i < zRow; i++) {
                for (std::size_t k = 0; k < xCol; k++) {
                    T xik = x_[i*xCol + k];
                    vecx = Vector::Func::set1(xik);
                    for (std::size_t j = 0; j < zCol - r; j+=Vector::N) {
                        /* put float into m256 */
                        vecy = Vector::Func::loadu(y_ + k*yCol + j);
                        vecz = Vector::Func::loadu(z_ + i*xCol + j);
                        /* _mm256_fmadd_ps(a, b, c): a*b + c */
                        Vector::Func::fmadd(vecx, vecy, vecz);
                        /* store result */
                        Vector::Func::storeu(z_ + i*xCol + j, vecz);
                    }
                    for (std::size_t j = zCol - r; j < zCol; j++) {
                        z_[i*zCol + j] += xik * y_[k*yCol + j];
                    }
                }
            }
            return;
        }

        inline static void kikj(T* __restrict z, std::size_t zRow, std::size_t zCol,
                                const T* __restrict x, std::size_t xRow, std::size_t xCol,
                                const T* __restrict y, std::size_t yRow, std::size_t yCol)
        {
            const T *x_ = x;
            const T *y_ = y;
            T *z_ = z;
            VectorType vecx;
            VectorType vecy;
            VectorType vecz;
            std::size_t r = zCol%Vector::N;
            for (std::size_t i = 0; i < zRow; i++) {
                for (std::size_t k = 0; k < xCol; k++) {
                    T xik = x_[i*xCol + k];
                    vecx = Vector::Func::set1(xik);
                    for (std::size_t j = 0; j < zCol - r; j+=Vector::N) {
                        /* put float into m256 */
                        vecy = Vector::Func::loadu(y_ + k*yCol + j);
                        vecz = Vector::Func::loadu(z_ + i*xCol + j);
                        /* _mm256_fmadd_ps(a, b, c): a*b + c */
                        Vector::Func::fmadd(vecx, vecy, vecz);
                        /* store result */
                        Vector::Func::storeu(z_ + i*xCol + j, vecz);
                    }
                    for (std::size_t j = zCol - r; j < zCol; j++) {
                        z_[i*zCol + j] += xik * y_[k*yCol + j];
                    }
                }
            }

            return;
        }
        inline static void kijk(T* __restrict z, std::size_t zRow, std::size_t zCol,
                                const T* __restrict x, std::size_t xRow, std::size_t xCol,
                                const T* __restrict y, std::size_t yRow, std::size_t yCol)
        {
            const T *x_ = x;
            const T *y_ = y;
            T *z_ = z;
            VectorType vecx;
            VectorType vecy;
            VectorType vecz;
            std::size_t r = zCol%Vector::N;
            for (std::size_t i = 0; i < zRow; i++) {
                for (std::size_t k = 0; k < xCol; k++) {
                    T xik = x_[i*xCol + k];
                    vecx = Vector::Func::set1(xik);
                    for (std::size_t j = 0; j < zCol - r; j+=Vector::N) {
                        /* put float into m256 */
                        vecy = Vector::Func::loadu(y_ + k*yCol + j);
                        vecz = Vector::Func::loadu(z_ + i*xCol + j);
                        /* _mm256_fmadd_ps(a, b, c): a*b + c */
                        Vector::Func::fmadd(vecx, vecy, vecz);
                        /* store result */
                        Vector::Func::storeu(z_ + i*xCol + j, vecz);
                    }
                    for (std::size_t j = zCol - r; j < zCol; j++) {
                        z_[i*zCol + j] += xik * y_[k*yCol + j];
                    }
                }
            }
            return;
        }
    };
};

}


#endif // SIMD_HPP
