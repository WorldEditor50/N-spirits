#include <iostream>
#include "../basic/tensorsi.hpp"
#include "../basic/simd.hpp"
#include "../basic/avx2func.hpp"
#include "../basic/util.hpp"
#include "../utils/clock.hpp"

float correct(const Tensorsi &x1, const Tensorsi &x2)
{
    float N = x1.totalSize;
    float n = 0;
    for (std::size_t i = 0; i < N; i++) {
        if (std::abs(x1[i] - x2[i]) < 1e-2) {
            n++;
        } else {
            //std::cout<<"error:"<<x1[i]<<" "<<x2[i]<<std::endl;
        }
    }
    return n/N*100;
}

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
    Tensorsi x(N, N);
    Tensorsi x1(N, N);
    Tensorsi x2(N, N);
    util::uniform(x1, -9, 9);
    util::uniform(x2, -9, 9);
    /* trivial matmul */
    {
        x.zero();
        Tensorsi x3(N, N);
        Tensorsi xt1(N, N);
        Tensorsi xt2(N, N);
        for (std::size_t i = 0; i < x1.totalSize; i++) {
            xt1[i] = x1[i];
            xt2[i] = x2[i];
        }
        auto t1 = Clock::tiktok();
        Tensorsi::Mul::ikkj(x3, xt1, xt2);
        auto t2 = Clock::tiktok();
        double cost = Clock::duration(t2, t1);
        std::cout<<"trivial matmul cost:"<<cost<<"s"<<std::endl;
        for (std::size_t i = 0; i < x.totalSize; i++) {
            x[i] = x3[i];
        }
        //x3.printValue();
    }
    /* simd matmul */
    /* 8-channel */
    {
        Tensorsi x3(N, N);
        auto t1 = Clock::tiktok();
        simd::AVX2::matMul(x3.ptr(), x3.shape[0], x3.shape[1],
                           x1.ptr(), x1.shape[0], x1.shape[1],
                           x2.ptr(), x2.shape[0], x2.shape[1]);
        auto t2 = Clock::tiktok();
        double cost = Clock::duration(t2, t1);
        std::cout<<"simd matmul8 cost:"<<cost<<"s"<<std::endl;
        std::cout<<"simd matmul8 correct rate:"<<correct(x, x3)<<"%"<<std::endl;
        //x3.printValue();
    }

    /* 64 channel */
    {
        Tensorsi x3(N, N);
        auto t1 = Clock::tiktok();
        simd::AVX2::matMul64(x3.ptr(), x3.shape[0], x3.shape[1],
                             x1.ptr(), x1.shape[0], x1.shape[1],
                             x2.ptr(), x2.shape[0], x2.shape[1]);
        auto t2 = Clock::tiktok();
        double cost = Clock::duration(t2, t1);
        std::cout<<"simd matmul64 cost:"<<cost<<"s"<<std::endl;
        std::cout<<"simd matmul64 correct rate:"<<correct(x, x3)<<"%"<<std::endl;
        //x3.printValue();
    }

    /* 8-channel wrapper */
    {
        Tensorsi x3(N, N);
        auto t1 = Clock::tiktok();
        simd::wrap<float, simd::AVX>::matMul(x3.ptr(), x3.shape[0], x3.shape[1],
                                             x1.ptr(), x1.shape[0], x1.shape[1],
                                             x2.ptr(), x2.shape[0], x2.shape[1]);
        auto t2 = Clock::tiktok();
        double cost = Clock::duration(t2, t1);
        std::cout<<"avx2 wrapper matmul cost:"<<cost<<"s"<<std::endl;
        std::cout<<"avx2 wrapper matmul correct rate:"<<correct(x, x3)<<"%"<<std::endl;
        //x3.printValue();
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
    Tensorsi x(100);
    util::uniform(x, 0, 100);
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
        util::uniform(x1, -1, 1);
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
        Tensorsi y1(100);
        x.fill(3);
        simd::AVX2::exp(y1.ptr(), x.ptr(), x.totalSize);
        std::cout<<"simd exp:"<<std::endl;
        y1.printValue();
        std::cout<<"cmath exp:"<<std::endl;
        Tensorsi y2(100);
        util::exp(x, y2);
        y2.printValue();
    }
    /* sqrt */
    {
        Tensorsi x1 = Tensorsi::ones(16, 16, 16);
        Tensorsi x2 = Tensorsi::ones(16, 16, 16);
        x1 += x2;
        Tensorsi x3 = Tensorsi::ones(16, 16, 16);
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
        Tensorsi x(N, N);
        Tensorsi y(N, N);
        util::uniform(x, 0, 9);
        auto t1 = Clock::tiktok();
        simd::AVX2::transpose(y.ptr(), N, N,
                              x.ptr(), N, N);
        auto t2 = Clock::tiktok();
        std::cout<<"avx2 transpose cost:"<<Clock::duration(t2, t1)<<"s"<<std::endl;
        t1 = Clock::tiktok();
        Tensorsi x1 = x.tr();
        t2 = Clock::tiktok();
        std::cout<<"trivial transpose cost:"<<Clock::duration(t2, t1)<<"s"<<std::endl;
    }
    if (0)
    {
        Tensorsi x(16, 16);
        Tensorsi y(16, 16);
        util::uniform(x, 0, 9);
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
#if defined(__SSE2__)
    std::cout<<"__m128/float:"<<sizeof (__m128)/sizeof (float)<<std::endl;
    std::cout<<"__m128d/double:"<<sizeof (__m128d)/sizeof (double)<<std::endl;
#elif defined (__AVX2__)
    std::cout<<"__m256/float:"<<sizeof (__m256)/sizeof (float)<<std::endl;
    std::cout<<"__m256d/double:"<<sizeof (__m256d)/sizeof (double)<<std::endl;
#endif
    //test_simd();
    //test_simd_matmul();
    //test_simd_transpose();
#if 1
    int N = 128;
    Tensorsi x(N, N);
    Tensorsi x1(N, N);
    Tensorsi x2(N, N);
    util::uniform(x1, -9, 9);
    util::uniform(x2, -9, 9);
    for (std::size_t i = 0; i < 4; i++) {
        x.zero();
        Tensorsi::Mul::ikkj(x, x1, x2);
    }
    Tensorsi x3(N, N);
    Tensorsi::Mul::ikkj(x3, x1, x2);
    std::cout<<"correct rate:"<<correct(x, x3)<<std::endl;
    x.printValue();
    x3.printValue();
#endif
	return 0;
}
