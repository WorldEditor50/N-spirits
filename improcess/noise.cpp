#include "noise.h"
#include "../basic/linalg.h"

Tensor imp::Noise::uniform(int h, int w, int c, float high, float low)
{
    Tensor x(h, w, c);
    LinAlg::uniform(x, high, low);
    return x;
}

Tensor imp::Noise::gaussian(int h, int w, int c, float sigma)
{
    Tensor x(h, w);
    std::uniform_real_distribution<float> roulette(0, 1);
    std::uniform_real_distribution<float> uniform(0, 1);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        float p = roulette(LinAlg::Random::engine);
        if (p > 0.01) {
            x[i] = 1;
            continue;
        }
        float x1 = uniform(LinAlg::Random::engine);
        float x2 = uniform(LinAlg::Random::engine);
        x[i] = sigma*std::log(x1)*std::cos(2*pi*x2);
    }
    return x;
}

Tensor imp::Noise::rayleigh(int h, int w, int c, float sigma)
{
    Tensor x(h, w, c);
    std::uniform_real_distribution<float> roulette(0, 1);
    std::uniform_real_distribution<float> uniform(0, 1);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        float p = roulette(LinAlg::Random::engine);
        if (p > 0.01) {
            x[i] = 1;
            continue;
        }
        float x1 = uniform(LinAlg::Random::engine);
        float x2 = uniform(LinAlg::Random::engine);
        float x3 = sigma*std::log(x1)*std::cos(2*pi*x2);
        float x4 = sigma*std::log(x2)*std::cos(2*pi*x1);
        x[i] = std::sqrt(x3*x3 + x4*x4);
    }
    return x;
}

Tensor imp::Noise::saltPepper(int h, int w, int c, float p0)
{
    Tensor x(h, w, c);
    std::uniform_real_distribution<float> roulette(0, 1);
    std::uniform_real_distribution<float> uniform(0, 1);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        float p = roulette(LinAlg::Random::engine);
        if (p < p0) {
            x[i] = uniform(LinAlg::Random::engine);
        } else {
            x[i] = 1;
        }
    }
    return x;
}
