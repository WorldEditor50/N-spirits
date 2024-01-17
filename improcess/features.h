#ifndef FEATURES_H
#define FEATURES_H
#include "improcess_def.h"

namespace imp {
/* histogram */
int histogram(OutTensor hist, InTensor gray);
int uniformHistogram(OutTensor hist, InTensor gray);
int moment0(OutTensor m0, InTensor hist);
int moment1(OutTensor m1, InTensor hist);
int entropy(InTensor img, int &thres);
int otsu(InTensor img, int &thres);
int grayConjugateMatrix(OutTensor xo, InTensor xi, const Point2i &p1, const Point2i &p2);
int barycenter(InTensor img, Point2i &center);

}
#endif // FEATURES_H
