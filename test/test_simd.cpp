#include <iostream>
#include "../basic/tensor.hpp"
#include "../basic/simd.hpp"
#include "../basic/avx2func.hpp"
#include "../basic/statistics.h"
#include "../utils/clock.hpp"

void test_simd_matmul()
{
#if defined(__AVX2__)
    std::cout<<"__m128/float:"<<sizeof (__m128)/sizeof (float)<<std::endl;
    std::cout<<"__m128d/double:"<<sizeof (__m128d)/sizeof (double)<<std::endl;
    std::cout<<"__m256/float:"<<sizeof (__m256)/sizeof (float)<<std::endl;
    std::cout<<"__m256d/double:"<<sizeof (__m256d)/sizeof (double)<<std::endl;

    /*

        N:       128         256         512        1024        2048
        cost:    0.000206s   0.002147s   0.014766s  0.211389s   2.58015s

    */
    int N = 2048;
    Tensor x(N, N);
    Tensor x1(N, N);
    Tensor x2(N, N);
    Statistics::uniform(x1, -9, 9);
    Statistics::uniform(x2, -9, 9);
    /* simd matmul */
    /* 8-channel */
    {
        auto t1 = Clock::tiktok();
        simd::AVX2::matMul(x.ptr(), x.shape[0], x.shape[1],
                           x1.ptr(), x1.shape[0], x1.shape[1],
                           x2.ptr(), x2.shape[0], x2.shape[1]);
        auto t2 = Clock::tiktok();
        double cost = Clock::duration(t2, t1);
        std::cout<<"simd matmul8 cost:"<<cost<<"s"<<std::endl;
        //x.printValue();
    }

    /* 64 channel */
    {
        x.zero();
        auto t1 = Clock::tiktok();
        simd::AVX2::matMul64(x.ptr(), x.shape[0], x.shape[1],
                             x1.ptr(), x1.shape[0], x1.shape[1],
                             x2.ptr(), x2.shape[0], x2.shape[1]);
        auto t2 = Clock::tiktok();
        double cost = Clock::duration(t2, t1);
        std::cout<<"simd matmul64 cost:"<<cost<<"s"<<std::endl;
        //x.printValue();
    }

    /* 8-channel wrapper */
    {
        x.zero();
        auto t1 = Clock::tiktok();
        simd::wrap<float, simd::AVX>::matMul(x.ptr(), x.shape[0], x.shape[1],
                                             x1.ptr(), x1.shape[0], x1.shape[1],
                                             x2.ptr(), x2.shape[0], x2.shape[1]);
        auto t2 = Clock::tiktok();
        double cost = Clock::duration(t2, t1);
        std::cout<<"avx2 wrapper matmul cost:"<<cost<<"s"<<std::endl;
        //x.printValue();
    }
    /* trivial matmul */
    {
        x.zero();
        auto t1 = Clock::tiktok();
        Tensor::Mul::ikkj(x, x1, x2);
        auto t2 = Clock::tiktok();
        double cost = Clock::duration(t2, t1);
        std::cout<<"trivial matmul cost:"<<cost<<"s"<<std::endl;
        //x.printValue();
    }
#endif
    return;
}
void test_simd()
{
#if defined(__AVX2__)
    /* horizontal sum */
    {
        __m256 x1 = _mm256_setr_ps(1, 1, 1, 1, 1, 1, 1, 1);
        float s1 = simd::AVX2::reduce(x1);
        std::cout<<"horizontal sum float = "<<s1<<std::endl;
        __m256d x2 = _mm256_setr_pd(1, 2, 3, 4);
        double s2 = simd::AVX2::reduce(x2);
        std::cout<<"horizontal sum double = "<<s2<<std::endl;
    }
    /* max */
    Tensor x(100);
    Statistics::uniform(x, 0, 100);
    x[50] = 101;
    x.printValue();
    std::cout<<"max:"<<simd::AVX2::max(x.ptr(), x.totalSize)<<std::endl;
    /* sum */
    Tensor x1(100, 100);
    x1.fill(1.0);
    std::cout<<"sum:"<<simd::AVX2::sum(x1.ptr(), x1.totalSize)<<std::endl;
    /* mean, variance */
    {
        Tensorsi_<float, simd::AVX2> x1(512, 512);
        Statistics::uniform(x1, -1, 1);
        float u = x1.mean();
        auto t1 = Clock::tiktok();
        float sigma = x1.variance(u);
        auto t2 = Clock::tiktok();
        double cost = Clock::duration(t2, t1);
        std::cout<<"cost time:"<<cost<<std::endl;
        std::cout<<"u:"<<u<<" sigma:"<<sigma<<std::endl;
    }
    /* exp */
    {
        Tensor y1(100);
        x.fill(3);
        simd::AVX2::exp(y1.ptr(), x.ptr(), x.totalSize);
        std::cout<<"simd exp:"<<std::endl;
        y1.printValue();
        std::cout<<"cmath exp:"<<std::endl;
        Tensor y2(100);
        Statistics::exp(x, y2);
        y2.printValue();
    }
    /* sqrt */
    {
        Tensor x1 = Tensor::ones(16, 16, 16);
        Tensor x2 = Tensor::ones(16, 16, 16);
        x1 += x2;
        Tensor x3 = Tensor::ones(16, 16, 16);
        simd::AVX2::sqrt(x3.ptr(), x1.ptr(), x1.totalSize);
        x3.printValue();
    }
#endif
    return;
}

void test_simd_transpose()
{
#if defined(__AVX2__)
    int N = 4096;
    {
        Tensor_<double> x(N, N);
        Tensor_<double> y(N, N);
        Statistics::uniform(x, 0, 9);
        auto t1 = Clock::tiktok();
        simd::AVX2::transpose(y.ptr(), N, N,
                              x.ptr(), N, N);
        auto t2 = Clock::tiktok();
        std::cout<<"avx2 transpose cost:"<<Clock::duration(t2, t1)<<"s"<<std::endl;
        t1 = Clock::tiktok();
        Tensor_<double> x1 = x.tr();
        t2 = Clock::tiktok();
        std::cout<<"trivial transpose cost:"<<Clock::duration(t2, t1)<<"s"<<std::endl;
    }
    if (0)
    {
        Tensor x(16, 16);
        Tensor y(16, 16);
        Statistics::uniform(x, 0, 9);
        simd::AVX2::transpose(y.ptr(), 16, 16,
                              x.ptr(), 16, 16);
        x.printValue();
        std::cout<<"simd transpose:"<<std::endl;
        y.printValue();
    }
    return;
#endif
}

int main()
{
    test_simd();
    test_simd_matmul();
    test_simd_transpose();
	return 0;
}
