#ifndef LINEARTRANSFORM_H
#define LINEARTRANSFORM_H
#include "image.hpp"

namespace imp {

/* histogram */
int histogram1(OutTensor hist, InTensor gray);
int histogram3(OutTensor hist, InTensor rgb);
/* transform */
int transform(OutTensor xo, InTensor xi, std::function<float(float)> func);
/* linear transform */
int linearTransform(OutTensor xo, InTensor xi, float alpha, float beta);
/* log transform */
int logTransform(OutTensor xo, InTensor xi, float c);
/* gamma transform */
int gammaTransform(OutTensor xo, InTensor xi, float exsp, float gamma);
/* threshold */
int threshold(OutTensor xo, InTensor xi, float thres, float max_, float min_);
/* histogram equalize */
int histogramEqualize(OutTensor xo, InTensor xi);

}

#endif // LINEARTRANSFORM_H
