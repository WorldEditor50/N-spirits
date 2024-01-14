#ifndef FEATURES_H
#define FEATURES_H
#include "improcess_def.h"

namespace imp {
/* histogram */
int histogram(OutTensor hist, InTensor gray);
int mean(InTensor hist, float &m);
int moment(InTensor hist, float m, int n, float &mu);
int entropy(InTensor hist, float &e);
int grayConjugateMatrix(OutTensor xo, InTensor xi, const Point2i &p1, const Point2i &p2);

}
#endif // FEATURES_H
