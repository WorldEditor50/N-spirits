#ifndef NOISE_H
#define NOISE_H
#include "image.hpp"
#include "../basic/util.hpp"

namespace Noise {
Tensor uniform(int h, int w, float high, float low);
Tensor gaussian(int h, int w, float sigma);
Tensor rayleigh(int h, int w, float sigma);
Tensor saltPepper(int h, int w);
}

#endif // NOISE_H
