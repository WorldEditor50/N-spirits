#ifndef GEOMETRYTRANSFORM_H
#define GEOMETRYTRANSFORM_H
#include "image.hpp"

namespace improcess {

/* geometry */
int move(const Tensor &x, const Point2i &offset, Tensor &y);
int transpose(const Tensor &x, Tensor &y);
int rotate(const Tensor &x, Tensor &y);
int flip(const Tensor &x, Tensor &y);
int resize(const Tensor &x, Tensor &y);
int interpolate(const Tensor &x, Tensor &y);
int affine();
int project();

}
#endif // GEOMETRYTRANSFORM_H
