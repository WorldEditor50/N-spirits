#ifndef AVX2WRAPPER_HPP
#define AVX2WRAPPER_HPP

#include <immintrin.h>
#include <vector>
#include <cmath>
#include "../basic_def.h"
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
       2. memory alignment is a easy way to make simd running in parallel
       3. use reference or pointer of variable will make function run faster
       4. use keyword __restrict, noexcept
       5. simd operations:
            a. load or set values
            b. process values
            b. store values

    Warning:
        1. this wrapper will work while the data size is greater than 8 for float

    Resource:
        1. https://www.cs.virginia.edu/~cr4bd/3330/F2018/simdref.html
*/


namespace simd {


#if defined(__AVX2__)

struct AVX {

struct M256d {
    using Type = __m256d;

    FORCE_INLINE static __m256d VECTORCALL load(double const* __restrict __p) noexcept
    {
        return _mm256_load_pd(__p);
    }
    FORCE_INLINE static __m256d VECTORCALL loadu(double const* __restrict __p) noexcept
    {
        return _mm256_loadu_pd(__p);
    }
    FORCE_INLINE static __m256d VECTORCALL set1(double& __restrict __w) noexcept
    {
        return _mm256_set1_pd(__w);
    }
    FORCE_INLINE static __m256d VECTORCALL zero() noexcept
    {
        return _mm256_setzero_pd();
    }
    FORCE_INLINE static void VECTORCALL store(double* __restrict __p, __m256d& __restrict __a) noexcept
    {
        return _mm256_store_pd(__p, __a);
    }
    FORCE_INLINE static void VECTORCALL storeu(double* __restrict __p, __m256d& __restrict __a) noexcept
    {
        return _mm256_storeu_pd(__p, __a);
    }
    FORCE_INLINE static __m256d VECTORCALL add(__m256d& __restrict __a, __m256d& __restrict __b) noexcept
    {
        return _mm256_add_pd(__a, __b);
    }
    FORCE_INLINE static __m256d VECTORCALL sub(__m256d& __restrict __a, __m256d& __restrict __b) noexcept
    {
        return _mm256_sub_pd(__a, __b);
    }
    FORCE_INLINE static __m256d VECTORCALL mul(__m256d& __restrict __a, __m256d& __restrict __b) noexcept
    {
        return _mm256_mul_pd(__a, __b);
    }
    FORCE_INLINE static __m256d VECTORCALL div(__m256d& __restrict __a, __m256d& __restrict __b) noexcept
    {
        return _mm256_div_pd(__a, __b);
    }
    FORCE_INLINE static __m256d VECTORCALL hadd(__m256d& __restrict __a, __m256d& __restrict __b) noexcept
    {
        return _mm256_hadd_pd(__a, __b);
    }
    FORCE_INLINE static __m256d VECTORCALL hsub(__m256d& __restrict __a, __m256d& __restrict __b) noexcept
    {
        return _mm256_hsub_pd(__a, __b);
    }
    FORCE_INLINE static void VECTORCALL fmadd(__m256d& __restrict __a, __m256d& __restrict __b, __m256d& __restrict __c) noexcept
    {
        __c = _mm256_fmadd_pd(__a, __b, __c);
    }
    FORCE_INLINE static void VECTORCALL fmsub(__m256d& __restrict __a, __m256d& __restrict __b, __m256d& __restrict __c) noexcept
    {
        __c = _mm256_fmsub_pd(__a, __b, __c);
        return;
    }

    FORCE_INLINE static __m256d VECTORCALL max(__m256d& __restrict __a, __m256d& __restrict __b) noexcept
    {
        return _mm256_max_pd(__a, __b);
    }
    FORCE_INLINE static __m256d VECTORCALL min(__m256d& __restrict __a, __m256d& __restrict __b) noexcept
    {
        return _mm256_min_pd(__a, __b);
    }
    FORCE_INLINE static __m256d VECTORCALL sqrt(__m256d& __restrict __a) noexcept
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
    using Type = __m256;
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

    FORCE_INLINE static __m256 VECTORCALL load(float const* __restrict __p) noexcept
    {
        return _mm256_load_ps(__p);
    }
    FORCE_INLINE static __m256 VECTORCALL loadu(float const* __restrict __p) noexcept
    {
        return _mm256_loadu_ps(__p);
    }
    FORCE_INLINE static __m256 VECTORCALL set1(float& __restrict __w) noexcept
    {
        return _mm256_set1_ps(__w);
    }
    FORCE_INLINE static __m256 VECTORCALL zero() noexcept
    {
        return _mm256_setzero_ps();
    }
    FORCE_INLINE static void VECTORCALL store(float* __restrict __p, __m256& __restrict __a) noexcept
    {
        return _mm256_store_ps(__p, __a);
    }
    FORCE_INLINE static void VECTORCALL storeu(float* __restrict __p, __m256& __restrict __a) noexcept
    {
        return _mm256_storeu_ps(__p, __a);
    }
    FORCE_INLINE static __m256 VECTORCALL add(__m256& __restrict __a, __m256& __restrict __b) noexcept
    {
        return _mm256_add_ps(__a, __b);
    }
    FORCE_INLINE static __m256 VECTORCALL sub(__m256& __restrict __a, __m256& __restrict __b) noexcept
    {
        return _mm256_sub_ps(__a, __b);
    }
    FORCE_INLINE static __m256 VECTORCALL mul(__m256& __restrict __a, __m256& __restrict __b) noexcept
    {
        return _mm256_mul_ps(__a, __b);
    }
    FORCE_INLINE static __m256 VECTORCALL div(__m256& __restrict __a, __m256& __restrict __b) noexcept
    {
        return _mm256_div_ps(__a, __b);
    }
    FORCE_INLINE static __m256 VECTORCALL hadd(__m256& __restrict __a, __m256& __restrict __b) noexcept
    {
        return _mm256_hadd_ps(__a, __b);
    }
    FORCE_INLINE static __m256 VECTORCALL hsub( __m256& __restrict __a, __m256& __restrict __b) noexcept
    {
        return _mm256_hsub_ps(__a, __b);
    }
    FORCE_INLINE static void VECTORCALL fmadd(__m256& __restrict __a, __m256& __restrict __b, __m256& __restrict __c) noexcept
    {
        __c =  _mm256_fmadd_ps(__a, __b, __c);
    }
    FORCE_INLINE static void VECTORCALL fmsub(__m256& __restrict __a, __m256& __restrict __b, __m256& __restrict __c) noexcept
    {
        __c = _mm256_fmsub_ps(__a, __b, __c);
    }

    FORCE_INLINE static __m256 VECTORCALL max(__m256& __restrict __a, __m256& __restrict __b) noexcept
    {
        return _mm256_max_ps(__a, __b);
    }
    FORCE_INLINE static __m256 VECTORCALL min(__m256& __restrict __a, __m256& __restrict __b) noexcept
    {
        return _mm256_min_ps(__a, __b);
    }
    FORCE_INLINE static __m256 VECTORCALL sqrt(__m256& __restrict __a) noexcept
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

};
#endif // AVX2

}

#endif // AVX2WRAPPER_HPP
