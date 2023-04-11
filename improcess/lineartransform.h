#ifndef LINEARTRANSFORM_H
#define LINEARTRANSFORM_H
#include "image.hpp"

namespace imp {

/* histogram */
int histogram1(const Tensor& gray, Tensor &hist);
int histogram3(const Tensor& rgb, Tensor &hist);
/* transform */
int transform(const Tensor &src, Tensor &dst, std::function<float(float)> func);
/* linear transform */
int linearTransform(const Tensor &x, float alpha, float beta, Tensor &y);
/* log transform */
int logTransform(const Tensor &x, float c, Tensor &y);
/* gamma transform */
int gammaTransform(const Tensor &x, float exsp, float gamma, Tensor &y);
/* threshold */
int threshold(const Tensor &x, float thres, float max_, float min_, Tensor &y);
/* histogram equalize */
int histogramEqualize(const Tensor &x, Tensor &y);

}

#endif // LINEARTRANSFORM_H
