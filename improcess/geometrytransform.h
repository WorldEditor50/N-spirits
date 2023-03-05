#ifndef GEOMETRYTRANSFORM_H
#define GEOMETRYTRANSFORM_H
#include "image.hpp"

namespace improcess {


/* geometry */
int move(const Tensor &src, Tensor &dst);
int rotate(const Tensor &src, Tensor &dst);
int flip(const Tensor &src, Tensor &dst);
int resize(const Tensor &src, Tensor &dst);
int interpolate(const Tensor &src, Tensor &dst);
int affine();
int project();

}
#endif // GEOMETRYTRANSFORM_H
