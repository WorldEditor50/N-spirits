#ifndef GRAPHIC3D_H
#define GRAPHIC3D_H
#include "../basic/tensor.hpp"
#include "../improcess/improcess_def.h"
#include "../improcess/color.h"

namespace imp {

class Camera
{
public:
    int width;
    int height;
    int *bitmap;
public:
    explicit Camera(int w, int h)
        :width(w),height(h)
    {
        bitmap = new int[h*w*4];
    }
    ~Camera()
    {
        if (bitmap != nullptr) {
            delete [] bitmap;
        }
    }

}; // camera

namespace Transform {

inline Tensor translate(float x, float y, float z)
{
    return Tensor({4, 4}, {1, 0, 0, 0,
                           0, 1, 0, 0,
                           0, 0, 1, 0,
                           x, y, z, 1});
}
inline Tensor scale(float sx, float sy, float sz)
{
    return Tensor({4, 4}, {sx, 0,  0,  0,
                           0,  sy, 0,  0,
                           0,  0,  sz, 0,
                           0,  0,  0,  1});
}

inline Tensor rotateX(float angle)
{
    float theta = angle*pi/180;
    float sinTheta = std::sin(theta);
    float cosTheta = std::cos(theta);
    return Tensor({4, 4}, {1,  0,         0,         0,
                           0,  cosTheta, -sinTheta,  0,
                           0,  sinTheta,  cosTheta,  0,
                           0,  0,         0,         1});
}

inline Tensor rotateY(float angle)
{
    float theta = angle*pi/180;
    float sinTheta = std::sin(theta);
    float cosTheta = std::cos(theta);
    return Tensor({4, 4}, {cosTheta, 0, sinTheta,   0,
                           0,        1, 0,          0,
                          -sinTheta, 0, cosTheta,   0,
                           0,        0, 0,          1});
}

inline Tensor rotateZ(float angle)
{
    float theta = angle*pi/180;
    float sinTheta = std::sin(theta);
    float cosTheta = std::cos(theta);
    return Tensor({4, 4}, {cosTheta, sinTheta, 0,   0,
                          -sinTheta, cosTheta, 0,   0,
                           0,        0,        1,   0,
                           0,        0,        0,   1});
}

}
Tensor ball(int x, int y, int z, float r, const Color3 &color);
int planeToSphere(OutTensor xo, InTensor xi, int r);
}

#endif // GRAPHIC3D_H
