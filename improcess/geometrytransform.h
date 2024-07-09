#ifndef GEOMETRYTRANSFORM_H
#define GEOMETRYTRANSFORM_H
#include "improcess_def.h"

namespace imp {

/* geometry */
int move(OutTensor xo, InTensor xi, const Size &offset);
int transpose(OutTensor xo, InTensor xi);
int horizontalFlip(OutTensor xo, InTensor xi);
int verticalFlip(OutTensor xo, InTensor xi);
int rotate(OutTensor xo, InTensor xi, float angle);
int nearestInterpolate(OutTensor xo, InTensor xi, const Size &size);
int bilinearInterpolate(OutTensor xo, InTensor xi, const Size &size);
int cubicInterpolate(OutTensor xo, InTensor xi, const Size &size, float a=0.5);
int affine(OutTensor xo, InTensor xi, InTensor op);

}
#endif // GEOMETRYTRANSFORM_H
