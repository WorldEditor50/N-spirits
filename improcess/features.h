#ifndef FEATURES_H
#define FEATURES_H
#include "improcess_def.h"

namespace imp {
/* histogram */
int histogram(OutTensor hist, InTensor gray);
int uniformHistogram(OutTensor hist, InTensor gray);
int moment0(OutTensor m0, InTensor hist);
int moment1(OutTensor m1, InTensor hist);
int entropy(InTensor img, uint8_t &thres);
int otsu(InTensor img, uint8_t &thres);
int grayConjugateMatrix(OutTensor xo, InTensor xi, const Point2i &p1, const Point2i &p2);
int barycenter(InTensor img, Point2i &center);
int LBP(OutTensor feature, InTensor gray);
int circleLBP(OutTensor feature, InTensor gray, int radius=3, int neighbors=8, bool rotationInvariant=true);
int multiScaleBlockLBP(OutTensor feature, InTensor gray, float scale=3);
}
#endif // FEATURES_H
