#ifndef NOISE_H
#define NOISE_H
#include "improcess_def.h"
#include "../basic/util.hpp"

namespace imp {
namespace Noise {

Tensor uniform(int h, int w, float high, float low);
Tensor gaussian(int h, int w, float sigma);
Tensor rayleigh(int h, int w, float sigma);
Tensor saltPepper(int h, int w);

}

}

#endif // NOISE_H
