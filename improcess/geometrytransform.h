#ifndef GEOMETRYTRANSFORM_H
#define GEOMETRYTRANSFORM_H
#include "image.hpp"

namespace imp {

/* geometry */
int move(const Tensor &x, const Point2i &offset, Tensor &y);
int transpose(const Tensor &x, Tensor &y);
int horizontalFlip(const Tensor &x, Tensor &y);
int verticalFlip(const Tensor &x, Tensor &y);
int scale(const Tensor &x, float alpha, Tensor &y);
int rotate(const Tensor &x, float angle, Tensor &y);
int bilinearInterpolate(const Tensor &x, Tensor &y);
int resize(const Tensor &x, Tensor &y);
int interpolate(const Tensor &x, Tensor &y);
int affine();
int project();

}
#endif // GEOMETRYTRANSFORM_H
