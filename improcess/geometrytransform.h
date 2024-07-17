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
namespace cubic {
    float triangle(float x);
    float bell(float x);
    float bspLine(float x);
}
int cubicInterpolate(OutTensor xo, InTensor xi, const Size &size, const std::function<float(float)> &interpolate);
namespace AffineOperator {
    /* translate */
    inline Tensor translate(float i, float j)
    {
        return Tensor({3, 3}, {1,   0,  0,
                               0,   1,  0,
                               i,  -j,  1});
    }
    /* scale */
    inline Tensor scale(float iFactor, float jFactor)
    {
        return Tensor({3, 3}, {iFactor, 0,       0,
                               0,       jFactor, 0,
                               0,       0,       1});
    }

    /* rotate */
    inline Tensor rotate(float angle)
    {
        float theta = angle*imp::pi/180.0;
        float sinTheta = std::sin(theta);
        float cosTheta = std::cos(theta);
        return Tensor({3, 3}, {cosTheta, sinTheta, 0,
                              -sinTheta, cosTheta, 0,
                               0,        0,        1});
    }
    /* shear in x direction */
    inline Tensor shearX(float factor)
    {
        return Tensor({3, 3}, {1,   factor, 0,
                               0,   1, 0,
                               0,   0, 1});
    }

    /* shear in y direction */
    inline Tensor shearY(float factor)
    {
        return Tensor({3, 3}, {1,      0, 0,
                               factor, 1, 0,
                               0,      0, 1});
    }
    /* reflect about x */
    inline Tensor flipX()
    {
        return Tensor({3, 3}, { 1,  0, 0,
                                0, -1, 0,
                                0,  0, 1});
    }
    /* reflect about y */
    inline Tensor flipY()
    {
        return Tensor({3, 3}, {-1,  0, 0,
                                0,  1, 0,
                                0,  0, 1});
    }
    /* operation center */
    inline Tensor center(float i, float j)
    {
        return Tensor({3, 3}, {1,  0, 0,
                               0, -1, 0,
                               j,  i, 1});
    }
}
int affine(OutTensor xo, InTensor xi, InTensor op);

}
#endif // GEOMETRYTRANSFORM_H
