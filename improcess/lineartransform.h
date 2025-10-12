#ifndef LINEARTRANSFORM_H
#define LINEARTRANSFORM_H
#include "../basic/tensor.hpp"
#include <functional>

namespace ns {
/* transform */
int transform(OutTensor xo, InTensor xi, std::function<float(float)> func);
/* linear transform */
int linearTransform(OutTensor xo, InTensor xi, float alpha, float beta);
/* log transform */
int logTransform(OutTensor xo, InTensor xi, float c);
/* gamma transform */
int gammaTransform(OutTensor xo, InTensor xi, float esp, float gamma);
/* histogram equalize */
int histogramEqualize(OutTensor xo, InTensor xi);
/* histogram standardize */
int histogramStandardize(OutTensor xo, InTensor xi);

}

#endif // LINEARTRANSFORM_H
