#ifndef SSE2FUNC_HPP
#define SSE2FUNC_HPP

#include <immintrin.h>
#include <vector>
#include <cmath>
#include <iostream>
#include "basic_def.h"

namespace simd {
#if defined(__SSE2__)
template<typename T>
struct Step {
    constexpr static std::size_t value = 0;
};
template<>
struct Step<double> {
    constexpr static std::size_t value = sizeof (__m128d)/sizeof (double);
};
template<>
struct Step<float> {
    constexpr static std::size_t value = sizeof (__m128)/sizeof (float);
};
struct SSE2 {

    inline static float reduce(__m128& ymm)
    {
        float result[4];
        //ymm = _mm_add_ps(ymm, ymm);
        //ymm = _mm_add_ps(ymm, ymm);
        //_mm_storeu_ps(result, ymm);
        return result[0] + result[3];
    }

    inline static double reduce(__m128d& ymm)
    {
        double result[2];
        //ymm = _mm_add_pd(ymm, ymm);
        //_mm_storeu_pd(result, ymm);
        return result[0] + result[1];
    }

    inline static void fill(double* __restrict x, double x0, std::size_t N)
    {
        double* px = x;
        std::size_t r = N%2;
        __m128d vecx0 = _mm_setr_pd(x0, x0);
        for (std::size_t i = 0; i < N - r; i+=2) {
            _mm_store_pd(px + i, vecx0);
        }
        for (std::size_t i = N - r; i < N; i++) {
            x[i] = x0;
        }
        return;
    }

    inline static void fill(float* __restrict x, float x0, std::size_t N)
    {
        float* px = x;
        std::size_t r = N%4;
        __m128 vecx0 = _mm_setr_ps(x0, x0, x0, x0);
        for (std::size_t i = 0; i < N - r; i+=4) {
            _mm_store_ps(px + i, vecx0);
        }
        for (std::size_t i = N - r; i < N; i++) {
            x[i] = x0;
        }
        return;
    }

    inline static void add(double* __restrict z, const double* __restrict y, const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        const double *py = y;
        double *pz = z;
        /* __m128d: sse double */
        __m128d vecx;
        __m128d vecy;
        __m128d vecz;
        /* step */
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            /* put 2 double into __m128d */
            vecx = _mm_loadu_pd(px + i);
            vecy = _mm_loadu_pd(py + i);
            /* add */
            vecz = _mm_add_pd(vecx, vecy);
            /* store result */
            _mm_storeu_pd(pz + i, vecz);
        }
        return;
    }

    inline static void sub(double* __restrict z, const double* __restrict y, const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        const double *py = y;
        double *pz = z;
        /* __m128d: sse double */
        __m128d vecx;
        __m128d vecy;
        __m128d vecz;
        /* offset */
        /* step */
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            /* put 2 double into __m128d */
            vecx = _mm_loadu_pd(px + i);
            vecy = _mm_loadu_pd(py + i);
            /* add */
            vecz = _mm_sub_pd(vecx, vecy);
            /* store result */
            _mm_storeu_pd(pz + i, vecz);
        }
        return;
    }

    inline static void mul(double* __restrict z, const double* __restrict y, const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        const double *py = y;
        double *pz = z;
        /* __m128d: sse double */
        __m128d vecx;
        __m128d vecy;
        __m128d vecz;
        /* step */
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            /* put 2 double into __m256d */
            vecx = _mm_loadu_pd(px + i);
            vecy = _mm_loadu_pd(py + i);
            /* add */
            vecz = _mm_mul_pd(vecx, vecy);
            /* store result */
            _mm_storeu_pd(pz + i, vecz);
        }
        return;
    }

    inline static void div(double* __restrict z, const double* __restrict y, const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        const double *py = y;
        double *pz = z;
        /* __m128d: sse double */
        __m128d vecx;
        __m128d vecy;
        __m128d vecz;
        /* step */
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            /* put 4 double into __m256d */
            vecx = _mm_loadu_pd(px + i);
            vecy = _mm_loadu_pd(py + i);
            /* add */
            vecz = _mm_div_pd(vecx, vecy);
            /* store result */
            _mm_storeu_pd(pz + i, vecz);
        }
        return;
    }

    inline static void add(float* __restrict z, const float* __restrict y, const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        const float *py = y;
        float *pz = z;
        /* __m128: sse2 float */
        __m128 vecx;
        __m128 vecy;
        __m128 vecz;
        /* step */
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            /* put 4 float into __m128 */
            vecx = _mm_loadu_ps(px + i);
            vecy = _mm_loadu_ps(py + i);
            /* add */
            vecz = _mm_add_ps(vecx, vecy);
            /* store result */
            _mm_storeu_ps(pz + i, vecz);
        }
        return;
    }

    inline static void sub(float* __restrict z, const float* __restrict y, const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        const float *py = y;
        float *pz = z;
        /* __m128: sse2 float */
        __m128 vecx;
        __m128 vecy;
        __m128 vecz;
        /* step */
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            /* put 4 float into __m128 */
            vecx = _mm_loadu_ps(px + i);
            vecy = _mm_loadu_ps(py + i);
            /* sub */
            vecz = _mm_sub_ps(vecx, vecy);
            /* store result */
            _mm_storeu_ps(pz + i, vecz);
        }
        return;
    }

    inline static void mul(float* __restrict z, const float* __restrict y, const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        const float *py = y;
        float *pz = z;
        /* __m128: sse2 float */
        __m128 vecx;
        __m128 vecy;
        __m128 vecz;
        /* step */
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            /* put 4 float into __m128 */
            vecx = _mm_loadu_ps(px + i);
            vecy = _mm_loadu_ps(py + i);
            /* add */
            vecz = _mm_mul_ps(vecx, vecy);
            /* store result */
            _mm_storeu_ps(pz + i, vecz);
        }
        return;
    }

    inline static void div(float* __restrict z, const float* __restrict y, const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        const float *py = y;
        float *pz = z;
        /* __m128: sse2 float */
        __m128 vecx;
        __m128 vecy;
        __m128 vecz;
        /* step */
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            /* put 4 double into __m256d */
            vecx = _mm_loadu_ps(px + i);
            vecy = _mm_loadu_ps(py + i);
            /* add */
            vecz = _mm_div_ps(vecx, vecy);
            /* store result */
            _mm_storeu_ps(pz + i, vecz);
        }
        return;
    }

    inline static void add(double* __restrict z, const double* __restrict y, double x, std::size_t N)
    {
        const double *py = y;
        double *pz = z;
        __m128d vecx = _mm_setr_pd(x, x);
        __m128d vecy;
        __m128d vecz;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            vecy = _mm_loadu_pd(py + i);
            vecz = _mm_add_pd(vecy, vecx);
            _mm_storeu_pd(pz + i, vecz);
        }
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = y[i] + x;
        }
        return;
    }

    inline static void sub(double* __restrict z, const double* __restrict y, double x, std::size_t N)
    {
        const double *py = y;
        double *pz = z;
        __m128d vecx = _mm_setr_pd(x, x);
        __m128d vecy;
        __m128d vecz;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            vecy = _mm_loadu_pd(py + i);
            vecz = _mm_sub_pd(vecy, vecx);
            _mm_storeu_pd(pz + i, vecz);
        }
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = y[i] - x;
        }
        return;
    }

    inline static void mul(double* __restrict z, const double* __restrict y, double x, std::size_t N)
    {
        const double *py = y;
        double *pz = z;
        __m128d vecx = _mm_setr_pd(x, x);
        __m128d vecy;
        __m128d vecz;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            vecy = _mm_loadu_pd(py + i);
            vecz = _mm_mul_pd(vecy, vecx);
            _mm_storeu_pd(pz + i, vecz);
        }
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = y[i] * x;
        }
        return;
    }

    inline static void div(double* __restrict z, const double* __restrict y, double x, std::size_t N)
    {
        const double *py = y;
        double *pz = z;
        __m128d vecx = _mm_setr_pd(x, x);
        __m128d vecy;
        __m128d vecz;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            vecy = _mm_loadu_pd(py + i);
            vecz = _mm_sub_pd(vecy, vecx);
            _mm_storeu_pd(pz + i, vecz);
        }
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = y[i] / x;
        }
        return;
    }

    inline static void add(float* __restrict z, const float* __restrict y, float x, std::size_t N)
    {
        const float *py = y;
        float *pz = z;
        __m128 vecx = _mm_setr_ps(x, x, x, x);
        __m128 vecy;
        __m128 vecz;
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            vecy = _mm_loadu_ps(py + i);
            vecz = _mm_add_ps(vecy, vecx);
            _mm_storeu_ps(pz + i, vecz);
        }
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = y[i] + x;
        }
        return;
    }

    inline static void sub(float* __restrict z, const float* __restrict y, float x, std::size_t N)
    {
        const float *py = y;
        float *pz = z;
        __m128 vecx = _mm_setr_ps(x, x, x, x);
        __m128 vecy;
        __m128 vecz;
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            vecy = _mm_loadu_ps(py + i);
            vecz = _mm_sub_ps(vecy, vecx);
            _mm_storeu_ps(pz + i, vecz);
        }
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = y[i] - x;
        }
        return;
    }

    inline static void mul(float* __restrict z, const float* __restrict y, float x, std::size_t N)
    {
        const float *py = y;
        float *pz = z;
        __m128 vecx = _mm_setr_ps(x, x, x, x);
        __m128 vecy;
        __m128 vecz;
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            vecy = _mm_loadu_ps(py + i);
            vecz = _mm_mul_ps(vecy, vecx);
            _mm_storeu_ps(pz + i, vecz);
        }
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = y[i] * x;
        }
        return;
    }

    inline static void div(float* __restrict z, const float* __restrict y, float x, std::size_t N)
    {
        const float *py = y;
        float *pz = z;
        __m128 vecx = _mm_setr_ps(x, x, x, x);
        __m128 vecy;
        __m128 vecz;
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = N%step;
        for (std::size_t i = 0; i < N - r; i+= step) {
            vecy = _mm_loadu_ps(py + i);
            vecz = _mm_div_ps(vecy, vecx);
            _mm_storeu_ps(pz + i, vecz);
        }
        for (std::size_t i = N - r; i < N; i++) {
            z[i] = y[i] / x;
        }
        return;
    }


    inline static double max(const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        /* align */
        std::size_t r = N%step;
        /* find max value per 2 element */
        __m128d maxVec = _mm_loadu_pd(px);
        for (std::size_t i = step; i < N - r; i+=step) {
            __m128d vecx = _mm_loadu_pd(px + i);
            maxVec = _mm_max_pd(vecx, maxVec);
        }
        /* find max value in result */
        double result[2];
        _mm_storeu_pd(result, maxVec);
        double maxValue = result[0];
        for (std::size_t i = 1; i < step; i++) {
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
        std::size_t step = sizeof (__m128)/sizeof (float);
        /* align */
        std::size_t r = N%step;
        /* find max value per 4 element */
        __m128 maxVec = _mm_loadu_ps(px);
        for (std::size_t i = step; i < N - r; i+=step) {
            __m128 vecx = _mm_loadu_ps(px + i);
            maxVec = _mm_max_ps(vecx, maxVec);
        }
        /* find max value in result */
        float result[4];
        _mm_storeu_ps(result, maxVec);
        float maxValue = result[0];
        for (std::size_t i = 1; i < step; i++) {
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
        std::size_t step = sizeof (__m128d)/sizeof (double);
        /* align */
        std::size_t r = N%step;
        /* find min value per 2 element */
        __m128d minVec = _mm_loadu_pd(px);
        for (std::size_t i = step; i < N - r; i+=step) {
            __m128d vecx = _mm_loadu_pd(px + i);
            minVec = _mm_min_pd(vecx, minVec);
        }
        /* find min value in result */
        double result[2];
        _mm_storeu_pd(result, minVec);
        double minValue = result[0];
        for (std::size_t i = 1; i < step; i++) {
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
        std::size_t step = sizeof (__m128)/sizeof (float);
        /* align */
        std::size_t r = N%step;
        /* find min value per 4 element */
        __m128 minVec = _mm_loadu_ps(px);
        for (std::size_t i = step; i < N - r; i+=step) {
            __m128 vecx = _mm_loadu_ps(px + i);
            minVec = _mm_min_ps(vecx, minVec);
        }
        /* find min value in result */
        float result[4];
        _mm_storeu_ps(result, minVec);
        float minValue = result[0];
        for (std::size_t i = 1; i < step; i++) {
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
        std::size_t step = sizeof (__m128d)/sizeof (double);
        /* init */
        std::size_t r = N%step;
        double result[2] = {0};
        __m128d vecs = _mm_loadu_pd(result);
        for (std::size_t i = 0; i < N - r; i+=step) {
            __m128d vecx = _mm_loadu_pd(px + i);
            vecs = _mm_add_pd(vecs, vecx);
        }
        /* sum up result */
        _mm_storeu_pd(result, vecs);
        double s = 0;
        for (std::size_t i = 0; i < step; i++) {
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
        std::size_t step = sizeof (__m128)/sizeof (float);
        /* init */
        std::size_t r = N%step;
        __m128 vecs = _mm_setzero_ps();
        for (std::size_t i = 0; i < N - r; i+=step) {
            __m128 vecx = _mm_loadu_ps(px + i);
            vecs = _mm_add_ps(vecs, vecx);
        }
        /* sum up result */
        float s = reduce(vecs);
        /* sum up the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            s += px[i];
        }
        return s;
    }
    inline static double product(const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        /* init */
        std::size_t r = N%step;
        double result[2] = {0};
        __m128d vecx;
        __m128d vecs = _mm_setr_pd(1, 1);
        for (std::size_t i = 0; i < N - r; i+=step) {
            vecx = _mm_loadu_pd(px + i);
            vecs = _mm_mul_pd(vecs, vecx);
        }
        float s = 1;
        _mm_storeu_pd(result, vecs);
        for (std::size_t i = 0; i < step; i++) {
            s *= result[i];
        }
        /* the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            s *= px[i];
        }
        return s;
    }

    inline static float product(const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        std::size_t step = sizeof (__m128)/sizeof (float);
        /* init */
        std::size_t r = N%step;
        float result[4] = {0};
        __m128 vecs = _mm_setr_ps(1, 1, 1, 1);
        for (std::size_t i = 0; i < N - r; i+=step) {
            __m128 vecx = _mm_loadu_ps(px + i);
            vecs = _mm_mul_ps(vecs, vecx);
        }
        float s = 1;
        _mm_storeu_ps(result, vecs);
        for (std::size_t i = 0; i < step; i++) {
            s *= result[i];
        }
        /* the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            s *= px[i];
        }
        return s;
    }

    inline static void sqrt(double* __restrict y, const double* __restrict x, std::size_t N)
    {
        const double *px = x;
        double *py = y;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        __m128d vecx;
        __m128d vecy;
        for (std::size_t i = 0; i < N - r; i+=step) {
            vecx = _mm_loadu_pd(px + i);
            vecy = _mm_sqrt_pd(vecx);
            _mm_storeu_pd(py + i, vecy);
        }
        /* sqrt the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            py[i] = std::sqrt(px[i]);
        }
        return;
    }

    inline static void sqrt(float* __restrict y, const float* __restrict x, std::size_t N)
    {
        const float *px = x;
        float *py = y;
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = N%step;
        __m128 vecx;
        __m128 vecy;
        for (std::size_t i = 0; i < N - r; i+=step) {
            vecx = _mm_loadu_ps(px + i);
            vecy = _mm_sqrt_ps(vecx);
            _mm_storeu_ps(py + i, vecy);
        }
        /* sqrt the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            py[i] = std::sqrt(px[i]);
        }
        return;
    }

    inline static double dot(const double* __restrict x1, const double* __restrict x2, std::size_t N)
    {
        const double *px1 = x1;
        const double *px2 = x2;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        __m128d vecx1;
        __m128d vecx2;
        __m128d vecy = _mm_setzero_pd();
        for (std::size_t i = 0; i < N - r; i+=step) {
            vecx1 = _mm_loadu_pd(px1 + i);
            vecx2 = _mm_loadu_pd(px2 + i);
            vecy = _mm_fmadd_pd(vecx1, vecx2, vecy);
        }
        /* sum up result */
        double s = reduce(vecy);
        /* dot the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            s += px1[i]*px2[i];
        }
        return s;
    }

    inline static float dot(const float* __restrict x1,
                           const float* __restrict x2,
                           std::size_t N)
    {
        const float *px1 = x1;
        const float *px2 = x2;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        __m128 vecx1;
        __m128 vecx2;
        __m128 vecy = _mm_setzero_ps();
        for (std::size_t i = 0; i < N - r; i+=step) {
            vecx1 = _mm_loadu_ps(px1 + i);
            vecx2 = _mm_loadu_ps(px2 + i);
            vecy = _mm_fmadd_ps(vecx1, vecx2, vecy);
        }
        /* sum up result */
        float s = reduce(vecy);
        /* dot the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            s += px1[i]*px2[i];
        }
        return s;
    }

    inline static double norm2s(const double* __restrict x1, const double* __restrict x2, std::size_t N)
    {
        const double *px1 = x1;
        const double *px2 = x2;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        __m128d vecx1;
        __m128d vecx2;
        __m128d vecx;
        __m128d vecy = _mm_setzero_pd();
        for (std::size_t i = 0; i < N - r; i+=step) {
            vecx1 = _mm_loadu_pd(px1 + i);
            vecx2 = _mm_loadu_pd(px2 + i);
            vecx = _mm_sub_pd(vecx1, vecx2);
            vecy = _mm_fmadd_pd(vecx, vecx, vecy);
        }
        /* sum up result */
        double s = reduce(vecy);
        /* the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            s += (px1[i] - px2[i])*(px1[i] - px2[i]);
        }
        return s;
    }

    inline static float norm2s(const float* __restrict x1, const float* __restrict x2, std::size_t N)
    {
        const float *px1 = x1;
        const float *px2 = x2;
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = N%step;
        __m128 vecx1;
        __m128 vecx2;
        __m128 vecx;
        __m128 vecy = _mm_setzero_ps();
        for (std::size_t i = 0; i < N - r; i+=step) {
            vecx1 = _mm_loadu_ps(px1 + i);
            vecx2 = _mm_loadu_ps(px2 + i);
            vecx = _mm_sub_ps(vecx1, vecx2);
            vecy = _mm_fmadd_ps(vecx, vecx, vecy);
        }
        /* sum up result */
        float s = reduce(vecy);
        /* the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            s += (px1[i] - px2[i])*(px1[i] - px2[i]);
        }
        return s;
    }

    inline static double variance(const double* __restrict x,  double  u, std::size_t N)
    {
        const double *px = x;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        std::size_t r = N%step;
        __m128d vecx1;
        __m128d vecu = _mm_set1_pd(u);
        __m128d vecx;
        __m128d vecy;
        for (std::size_t i = 0; i < N - r; i+=step) {
            vecx1 = _mm_loadu_pd(px + i);
            vecx = _mm_sub_pd(vecx1, vecu);
            vecy = _mm_fmadd_pd(vecx, vecx, vecy);
        }
        /* sum up result */
        double s = reduce(vecy);
        /* the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            s += (px[i] - u)*(px[i] - u);
        }
        return s/double(N);
    }

    inline static float variance(const float* __restrict x,  float  u, std::size_t N)
    {
        const float *px = x;
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = N%step;
        __m128 vecx1;
        __m128 vecu = _mm_set1_ps(u);
        __m128 vecx;
        __m128 vecy;
        for (std::size_t i = 0; i < N - r; i+=step) {
            vecx1 = _mm_loadu_ps(px + i);
            vecx = _mm_sub_ps(vecx1, vecu);
            vecy = _mm_fmadd_ps(vecx, vecx, vecy);
        }
        /* sum up result */
        float s = reduce(vecy);
        /* the rest elements */
        for (std::size_t i = N - r; i < N; i++) {
            s += (px[i] - u)*(px[i] - u);
        }
        return s/float(N);
    }

    inline static void matMul(double* __restrict z, std::size_t zRow, std::size_t zCol,
                              const double* __restrict x, std::size_t xRow, std::size_t xCol,
                              const double* __restrict y, std::size_t yRow, std::size_t yCol)
    {
        const double *x_ = x;
        const double *y_ = y;
        std::size_t step = sizeof (__m128d)/sizeof (double);
        double *z_ = z;
        std::size_t r = zCol%step;
        for (std::size_t i = 0; i < zRow; i++) {
            for (std::size_t k = 0; k < xCol; k++) {
                double xik = x_[i*xCol + k];
                __m128d vecx = _mm_set1_pd(xik);
                for (std::size_t j = 0; j < zCol - r; j+=step) {
                    __m128d vecy = _mm_loadu_pd(y_ + k*yCol + j);
                    __m128d vecz = _mm_loadu_pd(z_ + i*xCol + j);
                    /* _mm_fmadd_ps(a, b, c): a*b + c */
                    vecz = _mm_fmadd_pd(vecx, vecy, vecz);
                    /* store result */
                    _mm_storeu_pd(z_ + i*xCol + j, vecz);
                }
                for (std::size_t j = zCol - r; j < zCol; j++) {
                    z_[i*zCol + j] += xik * y_[k*yCol + j];
                }
            }
        }
        return;
    }

    inline static void matMul8(double* __restrict z, std::size_t zRow, std::size_t zCol,
                               const double* __restrict x, std::size_t xRow, std::size_t xCol,
                               const double* __restrict y, std::size_t yRow, std::size_t yCol)
    {
        const double *x_ = x;
        const double *y_ = y;
        std::size_t step = 4;
        double *z_ = z;
        std::size_t r1 = xCol%step;
        std::size_t r2 = zCol%step;
        double xik[4];
        __m128d vecx[4];
        __m128d vecy[4];
        __m128d vecz;
        for (std::size_t i = 0; i < zRow; i++) {
            for (std::size_t k = 0; k < xCol - r1; k+=step) {
                xik[0] = x_[i*xCol + k];
                xik[1] = x_[i*xCol + k + 1];
                xik[2] = x_[i*xCol + k + 2];
                xik[3] = x_[i*xCol + k + 3];
                vecx[0] = _mm_set1_pd(xik[0]);
                vecx[1] = _mm_set1_pd(xik[1]);
                vecx[2] = _mm_set1_pd(xik[2]);
                vecx[3] = _mm_set1_pd(xik[3]);
                for (std::size_t j = 0; j < zCol - r2; j+=step) {
                    /* put 2 double into __m128d */
                    vecy[0] = _mm_loadu_pd(y_ + k*yCol + j);
                    vecy[1] = _mm_loadu_pd(y_ + (k + 1)*yCol + j);
                    vecy[2] = _mm_loadu_pd(y_ + (k + 2)*yCol + j);
                    vecy[3] = _mm_loadu_pd(y_ + (k + 3)*yCol + j);
                    vecz  = _mm_loadu_pd(z_ + i*xCol + j);
                    /* _mm_fmadd_ps(a, b, c): a*b + c */
                    vecz = _mm_fmadd_pd(vecx[0], vecy[0], vecz);
                    vecz = _mm_fmadd_pd(vecx[1], vecy[1], vecz);
                    vecz = _mm_fmadd_pd(vecx[2], vecy[2], vecz);
                    vecz = _mm_fmadd_pd(vecx[3], vecy[3], vecz);
                    /* store result */
                    _mm_storeu_pd(z_ + i*xCol + j, vecz);
                }
                for (std::size_t j = zCol - r2; j < zCol; j++) {
                    z_[i*zCol + j] += xik[0] * y_[k*yCol + j];
                    z_[i*zCol + j] += xik[1] * y_[(k + 1)*yCol + j];
                    z_[i*zCol + j] += xik[2] * y_[(k + 2)*yCol + j];
                    z_[i*zCol + j] += xik[3] * y_[(k + 3)*yCol + j];
                }
            }
        }
        return;
    }

    inline static void matMul(float* __restrict z, std::size_t zRow, std::size_t zCol,
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
        std::size_t step = sizeof (__m128)/sizeof (float);
        std::size_t r = zCol%step;
        __m128 vecx;
        __m128 vecy;
        __m128 vecz;
        for (std::size_t i = 0; i < zRow; i++) {
            for (std::size_t k = 0; k < xCol; k++) {
                float xik = x_[i*xCol + k];
                vecx = _mm_set1_ps(xik);
                for (std::size_t j = 0; j < zCol - r; j+=step) {
                    /* put 4 float into __m128 */
                    vecy = _mm_loadu_ps(y_ + k*yCol + j);
                    vecz = _mm_loadu_ps(z_ + i*xCol + j);
                    /* _mm_fmadd_ps(a, b, c): a*b + c */
                    vecz = _mm_fmadd_ps(vecx, vecy, vecz);
                    /* store result */
                    _mm_storeu_ps(z_ + i*xCol + j, vecz);
                }
                /* the rest column */
                for (std::size_t j = zCol - r; j < zCol; j++) {
                    z_[i*zCol + j] += xik * y_[k*yCol + j];
                }
            }
        }
        return;
    }

    inline static void matMul32(float* __restrict z, std::size_t zRow, std::size_t zCol,
                                const float* __restrict x, std::size_t xRow, std::size_t xCol,
                                const float* __restrict y, std::size_t yRow, std::size_t yCol)
    {
        const float *x_ = x;
        const float *y_ = y;
        float *z_ = z;
        std::size_t step = 8;
        std::size_t r1 = xCol%step;
        std::size_t r2 = zCol%step;
        float xik[8];
        __m128 vecx[8];
        __m128 vecy[8];
        __m128 vecz;
        for (std::size_t i = 0; i < zRow; i++) {
            for (std::size_t k = 0; k < xCol - r1; k+=step) {
                xik[0] = x_[i*xCol + k];
                xik[1] = x_[i*xCol + k + 1];
                xik[2] = x_[i*xCol + k + 2];
                xik[3] = x_[i*xCol + k + 3];
                xik[4] = x_[i*xCol + k + 4];
                xik[5] = x_[i*xCol + k + 5];
                xik[6] = x_[i*xCol + k + 6];
                xik[7] = x_[i*xCol + k + 7];
                vecx[0] = _mm_set1_ps(xik[0]);
                vecx[1] = _mm_set1_ps(xik[1]);
                vecx[2] = _mm_set1_ps(xik[2]);
                vecx[3] = _mm_set1_ps(xik[3]);
                vecx[4] = _mm_set1_ps(xik[4]);
                vecx[5] = _mm_set1_ps(xik[5]);
                vecx[6] = _mm_set1_ps(xik[6]);
                vecx[7] = _mm_set1_ps(xik[7]);
                for (std::size_t j = 0; j < zCol - r2; j+=step) {
                    /* put 4 float into __m128 */
                    vecy[0] = _mm_loadu_ps(y_ + k*yCol + j);
                    vecy[1] = _mm_loadu_ps(y_ + (k + 1)*yCol + j);
                    vecy[2] = _mm_loadu_ps(y_ + (k + 2)*yCol + j);
                    vecy[3] = _mm_loadu_ps(y_ + (k + 3)*yCol + j);
                    vecy[4] = _mm_loadu_ps(y_ + (k + 4)*yCol + j);
                    vecy[5] = _mm_loadu_ps(y_ + (k + 5)*yCol + j);
                    vecy[6] = _mm_loadu_ps(y_ + (k + 6)*yCol + j);
                    vecy[7] = _mm_loadu_ps(y_ + (k + 7)*yCol + j);
                    vecz  = _mm_loadu_ps(z_ + i*xCol + j);
                    /* _mm_fmadd_ps(a, b, c): a*b + c */
                    vecz = _mm_fmadd_ps(vecx[0], vecy[0], vecz);
                    vecz = _mm_fmadd_ps(vecx[1], vecy[1], vecz);
                    vecz = _mm_fmadd_ps(vecx[2], vecy[2], vecz);
                    vecz = _mm_fmadd_ps(vecx[3], vecy[3], vecz);
                    vecz = _mm_fmadd_ps(vecx[4], vecy[4], vecz);
                    vecz = _mm_fmadd_ps(vecx[5], vecy[5], vecz);
                    vecz = _mm_fmadd_ps(vecx[6], vecy[6], vecz);
                    vecz = _mm_fmadd_ps(vecx[7], vecy[7], vecz);
                    /* store result */
                    _mm_storeu_ps(z_ + i*xCol + j, vecz);
                }

                for (std::size_t j = zCol - r2; j < zCol; j++) {
                    z_[i*zCol + j] += xik[0] * y_[k*yCol + j];
                    z_[i*zCol + j] += xik[1] * y_[(k + 1)*yCol + j];
                    z_[i*zCol + j] += xik[2] * y_[(k + 2)*yCol + j];
                    z_[i*zCol + j] += xik[3] * y_[(k + 3)*yCol + j];
                    z_[i*zCol + j] += xik[4] * y_[(k + 4)*yCol + j];
                    z_[i*zCol + j] += xik[5] * y_[(k + 5)*yCol + j];
                    z_[i*zCol + j] += xik[6] * y_[(k + 6)*yCol + j];
                    z_[i*zCol + j] += xik[7] * y_[(k + 7)*yCol + j];
                }
            }
        }
        return;
    }

    struct MatMul {

        inline static void ikkj(float* __restrict z, std::size_t zRow, std::size_t zCol,
                                const float* __restrict x, std::size_t xRow, std::size_t xCol,
                                const float* __restrict y, std::size_t yRow, std::size_t yCol)
        {
            const float *x_ = x;
            const float *y_ = y;
            float *z_ = z;
            std::size_t step = sizeof (__m128)/sizeof (float);
            std::size_t r = zCol%step;
            __m128 vecx;
            __m128 vecy;
            __m128 vecz;
            for (std::size_t i = 0; i < zRow; i++) {
                for (std::size_t k = 0; k < xCol; k++) {
                    float xik = x_[i*xCol + k];
                    vecx = _mm_set1_ps(xik);
                    for (std::size_t j = 0; j < zCol - r; j+=step) {
                        /* put 4 float into __m128 */
                        vecy = _mm_loadu_ps(y_ + k*yCol + j);
                        vecz = _mm_loadu_ps(z_ + i*xCol + j);
                        /* _mm_fmadd_ps(a, b, c): a*b + c */
                        vecz = _mm_fmadd_ps(vecx, vecy, vecz);
                        /* store result */
                        _mm_storeu_ps(z_ + i*xCol + j, vecz);
                    }
                    for (std::size_t j = zCol - r; j < zCol; j++) {
                        z_[i*xCol + j] += xik * y_[k*yCol + j];
                    }
                }
            }
            return;
        }

        inline static void kikj(float* __restrict z, std::size_t zRow, std::size_t zCol,
                                const float* __restrict x, std::size_t xRow, std::size_t xCol,
                                const float* __restrict y, std::size_t yRow, std::size_t yCol)
        {
            /* z = x^T * y */
            const float *x_ = x;
            const float *y_ = y;
            float *z_ = z;
            std::size_t step = sizeof (__m128)/sizeof (float);
            std::size_t r = zCol%step;
            __m128 vecx;
            __m128 vecy;
            __m128 vecz;
            for (std::size_t i = 0; i < zRow; i++) {
                for (std::size_t k = 0; k < xRow; k++) {
                    float xki = x_[k*xCol + i];
                    vecx = _mm_set1_ps(xki);
                    for (std::size_t j = 0; j < zCol - r; j+=step) {
                        /* put 4 float into __m128 */
                        vecy = _mm_loadu_ps(y_ + k*yCol + j);
                        vecz = _mm_loadu_ps(z_ + i*xCol + j);
                        /* _mm_fmadd_ps(a, b, c): a*b + c */
                        vecz = _mm_fmadd_ps(vecx, vecy, vecz);
                        /* store result */
                        _mm_storeu_ps(z_ + i*xCol + j, vecz);
                    }
                    for (std::size_t j = zCol - r; j < zCol; j++) {
                        z_[i*zCol + j] += xki * y_[k*yCol + j];
                    }
                }
            }
            return;
        }
        inline static void ikjk(float* __restrict z, std::size_t zRow, std::size_t zCol,
                                const float* __restrict x, std::size_t xRow, std::size_t xCol,
                                const float* __restrict y, std::size_t yRow, std::size_t yCol)
        {
            /* z = x * y^T */
            const float *x_ = x;
            const float *y_ = y;
            float *z_ = z;
            std::size_t step = sizeof (__m128)/sizeof (float);
            std::size_t r = zCol%step;
            __m128 vecx;
            __m128 vecy;
            __m128 vecz;
            for (std::size_t i = 0; i < zRow; i++) {
                for (std::size_t k = 0; k < xCol; k++) {
                    float xik = x_[i*xCol + k];
                    vecx = _mm_set1_ps(xik);
                    for (std::size_t j = 0; j < zCol - r; j+=step) {
                        /* put 4 float into __m128 */
                        vecy = _mm_loadu_ps(y_ + k*yCol + j);
                        vecz = _mm_loadu_ps(z_ + i*xCol + j);
                        /* _mm_fmadd_ps(a, b, c): a*b + c */
                        vecz = _mm_fmadd_ps(vecx, vecy, vecz);
                        /* store result */
                        _mm_storeu_ps(z_ + i*xCol + j, vecz);
                    }
                    for (std::size_t j = zCol - r; j < zCol; j++) {
                        z_[i*zCol + j] += xik * y_[j*yCol + k];
                    }
                }
            }
            return;
        }
        inline static void kijk(float* __restrict z, std::size_t zRow, std::size_t zCol,
                                const float* __restrict x, std::size_t xRow, std::size_t xCol,
                                const float* __restrict y, std::size_t yRow, std::size_t yCol)
        {
            /* z = x^T * y^T */
            const float *x_ = x;
            const float *y_ = y;
            float *z_ = z;
            std::size_t step = sizeof (__m128)/sizeof (float);
            std::size_t r = zCol%step;
            __m128 vecx;
            __m128 vecy;
            __m128 vecz;
            for (std::size_t i = 0; i < zRow; i++) {
                for (std::size_t k = 0; k < xRow; k++) {
                    float xki = x_[k*xCol + i];
                    vecx = _mm_set1_ps(xki);
                    for (std::size_t j = 0; j < zCol - r; j+=step) {
                        /* put 4 float into __m128 */
                        vecy = _mm_loadu_ps(y_ + k*yCol + j);
                        vecz = _mm_loadu_ps(z_ + i*xCol + j);
                        /* _mm_fmadd_ps(a, b, c): a*b + c */
                        vecz = _mm_fmadd_ps(vecx, vecy, vecz);
                        /* store result */
                        _mm_storeu_ps(z_ + i*xCol + j, vecz);
                    }
                    for (std::size_t j = zCol - r; j < zCol; j++) {
                        z_[i*zCol + j] += xki * y_[j*yCol + k];
                    }
                }
            }
            return;
        }

    };
};

#endif // SSE2
}

#endif // SSE2FUNC_HPP
