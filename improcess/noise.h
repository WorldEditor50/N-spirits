#ifndef NOISE_H
#define NOISE_H
#include "improcess_def.h"

namespace imp {
namespace Noise {

Tensor uniform(int h, int w, int c, float high, float low);
Tensor gaussian(int h, int w, int c, float sigma);
Tensor rayleigh(int h, int w, int c, float sigma);
Tensor saltPepper(int h, int w, int c, float p0);

}

}

#endif // NOISE_H
