#ifndef LINEARTRANSFORM_H
#define LINEARTRANSFORM_H
#include "improcess_def.h"
#include "features.h"

namespace imp {
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
