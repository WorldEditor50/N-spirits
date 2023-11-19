#include "noise.h"


Tensor Noise::uniform(int h, int w, float high, float low)
{
    Tensor x(h, w);
    util::uniform(x, high, low);
    return x;
}

Tensor Noise::gaussian(int h, int w, float sigma)
{
    Tensor x(h, w);
    std::default_random_engine engine;
    std::uniform_real_distribution<float> p(0, 1);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        float x1 = p(engine);
        float x2 = p(engine);
        x[i] = sigma*std::log(x1)*std::cos(2*pi*x2);
    }
    return x;
}

Tensor Noise::rayleigh(int h, int w, float sigma)
{
    Tensor x(h, w);
    std::default_random_engine engine;
    std::uniform_real_distribution<float> p(0, 1);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        float x1 = p(engine);
        float x2 = p(engine);
        float x3 = sigma*std::log(x1)*std::cos(2*pi*x2);
        float x4 = sigma*std::log(x2)*std::cos(2*pi*x1);
        x[i] = std::sqrt(x3*x3 + x4*x4);
    }
    return x;
}

Tensor Noise::saltPepper(int h, int w)
{
    Tensor x(h, w);
    util::uniform(x, -255, 255);
    return x;
}
