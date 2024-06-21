#ifndef SSE2WRAPPER_HPP
#define SSE2WRAPPER_HPP

#include <vector>
#include <cmath>
#include <immintrin.h>
#include "../basic_def.h"

namespace simd {


#if defined(__SSE2__)

struct SSE {

struct M128d {

    using Type = __m128d;
    FORCE_INLINE static __m128d VECTORCALL load(double const* __restrict __p) noexcept
    {
        return _mm_load_pd(__p);
    }
    FORCE_INLINE static __m128d VECTORCALL loadu(double const* __restrict __p) noexcept
    {
        return _mm_loadu_pd(__p);
    }
    FORCE_INLINE static __m128d VECTORCALL set1(double& __restrict __w) noexcept
    {
        return _mm_set1_pd(__w);
    }
    FORCE_INLINE static __m128d VECTORCALL zero() noexcept
    {
        return _mm_setzero_pd();
    }
    FORCE_INLINE static void VECTORCALL store(double* __restrict __p, __m128d& __restrict __a) noexcept
    {
        return _mm_store_pd(__p, __a);
    }
    FORCE_INLINE static void VECTORCALL storeu(double* __restrict __p, __m128d& __restrict __a) noexcept
    {
        return _mm_storeu_pd(__p, __a);
    }
    FORCE_INLINE static __m128d VECTORCALL add(__m128d& __restrict __a, __m128d& __restrict __b) noexcept
    {
        return _mm_add_pd(__a, __b);
    }
    FORCE_INLINE static __m128d VECTORCALL sub(__m128d& __restrict __a, __m128d& __restrict __b) noexcept
    {
        return _mm_sub_pd(__a, __b);
    }
    FORCE_INLINE static __m128d VECTORCALL mul(__m128d& __restrict __a, __m128d& __restrict __b) noexcept
    {
        return _mm_mul_pd(__a, __b);
    }
    FORCE_INLINE static __m128d VECTORCALL div(__m128d& __restrict __a, __m128d& __restrict __b) noexcept
    {
        return _mm_div_pd(__a, __b);
    }
    FORCE_INLINE static __m128d VECTORCALL hadd(__m128d& __restrict __a, __m128d& __restrict __b) noexcept
    {
        return _mm_hadd_pd(__a, __b);
    }
    FORCE_INLINE static __m128d VECTORCALL hsub( __m128d& __restrict __a, __m128d& __restrict __b) noexcept
    {
        return _mm_hsub_pd(__a, __b);
    }
    FORCE_INLINE static void VECTORCALL fmadd(__m128d& __restrict __a, __m128d& __restrict __b, __m128d& __restrict __c) noexcept
    {
        __c =  _mm_fmadd_pd(__a, __b, __c);
    }
    FORCE_INLINE static void VECTORCALL fmsub(__m128d& __restrict __a, __m128d& __restrict __b, __m128d& __restrict __c) noexcept
    {
        __c = _mm_fmsub_pd(__a, __b, __c);
    }

    FORCE_INLINE static __m128d VECTORCALL max(__m128d& __restrict __a, __m128d& __restrict __b) noexcept
    {
        return _mm_max_pd(__a, __b);
    }
    FORCE_INLINE static __m128d VECTORCALL min(__m128d& __restrict __a, __m128d& __restrict __b) noexcept
    {
        return _mm_min_pd(__a, __b);
    }
    FORCE_INLINE static __m128d VECTORCALL sqrt(__m128d& __restrict __a) noexcept
    {
        return _mm_sqrt_pd(__a);
    }
    /* rewrap */
    using Add = __m128d (*)(__m128d& __restrict , __m128d& __restrict );
    using Sub = __m128d (*)(__m128d& __restrict , __m128d& __restrict );
    using Mul = __m128d (*)(__m128d& __restrict , __m128d& __restrict );
    using Div = __m128d (*)(__m128d& __restrict , __m128d& __restrict );
    using Max = __m128d (*)(__m128d& __restrict , __m128d& __restrict );
    using Min = __m128d (*)(__m128d& __restrict , __m128d& __restrict );

};

struct M128 {
    using Type = __m128;
    FORCE_INLINE static __m128 VECTORCALL load(float const* __restrict __p) noexcept
    {
        return _mm_load_ps(__p);
    }
    FORCE_INLINE static __m128 VECTORCALL loadu(float const* __restrict __p) noexcept
    {
        return _mm_loadu_ps(__p);
    }
    FORCE_INLINE static __m128 VECTORCALL set1(float& __restrict __w) noexcept
    {
        return _mm_set1_ps(__w);
    }
    FORCE_INLINE static __m128 VECTORCALL zero() noexcept
    {
        return _mm_setzero_ps();
    }
    FORCE_INLINE static void VECTORCALL store(float* __restrict __p, __m128& __restrict __a) noexcept
    {
        return _mm_store_ps(__p, __a);
    }
    FORCE_INLINE static void VECTORCALL storeu(float* __restrict __p, __m128& __restrict __a) noexcept
    {
        return _mm_storeu_ps(__p, __a);
    }
    FORCE_INLINE static __m128 VECTORCALL add(__m128& __restrict __a, __m128& __restrict __b) noexcept
    {
        return _mm_add_ps(__a, __b);
    }
    FORCE_INLINE static __m128 VECTORCALL sub(__m128& __restrict __a, __m128& __restrict __b) noexcept
    {
        return _mm_sub_ps(__a, __b);
    }
    FORCE_INLINE static __m128 VECTORCALL mul(__m128& __restrict __a, __m128& __restrict __b) noexcept
    {
        return _mm_mul_ps(__a, __b);
    }
    FORCE_INLINE static __m128 VECTORCALL div(__m128& __restrict __a, __m128& __restrict __b) noexcept
    {
        return _mm_div_ps(__a, __b);
    }
    FORCE_INLINE static __m128 VECTORCALL hadd(__m128& __restrict __a, __m128& __restrict __b) noexcept
    {
        return _mm_hadd_ps(__a, __b);
    }
    FORCE_INLINE static __m128 VECTORCALL hsub( __m128& __restrict __a, __m128& __restrict __b) noexcept
    {
        return _mm_hsub_ps(__a, __b);
    }
    FORCE_INLINE static void VECTORCALL fmadd(__m128& __restrict __a, __m128& __restrict __b, __m128& __restrict __c) noexcept
    {
        __c =  _mm_fmadd_ps(__a, __b, __c);
    }
    FORCE_INLINE static void VECTORCALL fmsub(__m128& __restrict __a, __m128& __restrict __b, __m128& __restrict __c) noexcept
    {
        __c = _mm_fmsub_ps(__a, __b, __c);
    }

    FORCE_INLINE static __m128 VECTORCALL max(__m128& __restrict __a, __m128& __restrict __b) noexcept
    {
        return _mm_max_ps(__a, __b);
    }
    FORCE_INLINE static __m128 VECTORCALL min(__m128& __restrict __a, __m128& __restrict __b) noexcept
    {
        return _mm_min_ps(__a, __b);
    }
    FORCE_INLINE static __m128 VECTORCALL sqrt(__m128& __restrict __a) noexcept
    {
        return _mm_sqrt_ps(__a);
    }
    /* rewrap */
    using Add = __m128 (*)(__m128& __restrict , __m128& __restrict );
    using Sub = __m128 (*)(__m128& __restrict , __m128& __restrict );
    using Mul = __m128 (*)(__m128& __restrict , __m128& __restrict );
    using Div = __m128 (*)(__m128& __restrict , __m128& __restrict );
    using Max = __m128 (*)(__m128& __restrict , __m128& __restrict );
    using Min = __m128 (*)(__m128& __restrict , __m128& __restrict );

};

FORCE_INLINE double VECTORCALL reduce(const double * const adr)
{
    double s;
    __m128d low = _mm_loadu_pd(&adr[0]);
    __m128d high = _mm_loadu_pd(&adr[2]);
    __m128d hsum = _mm_add_pd(low, high);
    hsum = _mm_hadd_pd(hsum, hsum);
    _mm_store_sd(&s, hsum);
    return s;
}

FORCE_INLINE float VECTORCALL reduce(const float * const adr)
{
    float s;
    __m128 low = _mm_loadu_ps(&adr[0]);
    __m128 high = _mm_loadu_ps(&adr[4]);
    __m128 hsum = _mm_add_ps(low, high);
    hsum = _mm_hadd_ps(hsum, hsum);
    hsum = _mm_hadd_ps(hsum, hsum);
    _mm_store_ss(&s, hsum);
    return s;
}

};
#endif // SSE


} // simd

#endif // SSE2WRAPPER_HPP
