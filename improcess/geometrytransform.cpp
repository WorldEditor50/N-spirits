#include "geometrytransform.h"

int imp::move(OutTensor xo, InTensor xi, const Size &offset)
{
    if (offset.x < 0 || offset.x > xi.shape[HWC_W] ||
        offset.y < 0 || offset.y > xi.shape[HWC_H]) {
        return -1;
    }
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    int c = xi.shape[HWC_C];
    xo = Tensor(h + offset.x, w + offset.y, c);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            if (i < offset.y || j < offset.x) {
                continue;
            }
            for (int k = 0; k < c; k++) {
                xo(i, j, k) = xi(i - offset.y, j - offset.x, k);
            }
        }
    }
    return 0;
}

int imp::transpose(OutTensor xo, InTensor xi)
{
    /* HWC -> WHC */
    xo = xi.permute(1, 0, 2);
    return 0;
}

int imp::horizontalFlip(OutTensor xo, InTensor xi)
{
    int w = xi.shape[HWC_W];
    xo = Tensor(xi.shape);
    for (int i = 0; i < xi.shape[HWC_H]; i++) {
        for (int j = 0; j < xi.shape[HWC_W]; j++) {
            for (int k = 0; k < xi.shape[HWC_C]; k++) {
                xo(i, j, k) = xi(i, w - j - 1, k);
            }
        }
    }
    return 0;
}

int imp::verticalFlip(OutTensor xo, InTensor xi)
{
    int h = xi.shape[HWC_H];
    xo = Tensor(xi.shape);
    for (int i = 0; i < xi.shape[HWC_H]; i++) {
        for (int j = 0; j < xi.shape[HWC_W]; j++) {
            for (int k = 0; k < xi.shape[HWC_C]; k++) {
                xo(i, j, k) = xi(h - i - 1, j, k);
            }
        }
    }
    return 0;
}

int imp::rotate(OutTensor xo, InTensor xi, float angle)
{
    /* rotate center = (h/2, w/2) */
    float theta = angle*pi/180.0;
    float sinTheta = std::sin(theta);
    float cosTheta = std::cos(theta);
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    int c = xi.shape[HWC_C];

    /* original image's corners */
    float wi1 = -(w - 1)*0.5;
    float hi1 =  (h - 1)*0.5;
    float wi2 =  (w - 1)*0.5;
    float hi2 =  (h - 1)*0.5;
    float wi3 = -(w - 1)*0.5;
    float hi3 = -(h - 1)*0.5;
    float wi4 =  (w - 1)*0.5;
    float hi4 = -(h - 1)*0.5;
    /* output image's corners */
    float wo1 =  cosTheta*wi1 + sinTheta*hi1;
    float ho1 = -sinTheta*wi1 + cosTheta*hi1;
    float wo2 =  cosTheta*wi2 + sinTheta*hi2;
    float ho2 = -sinTheta*wi2 + cosTheta*hi2;
    float wo3 =  cosTheta*wi3 + sinTheta*hi3;
    float ho3 = -sinTheta*wi3 + cosTheta*hi3;
    float wo4 =  cosTheta*wi4 + sinTheta*hi4;
    float ho4 = -sinTheta*wi4 + cosTheta*hi4;
    /* output size */
    int wo = float(std::max(std::abs(wo4 - wo1), std::abs(wo3 - wo2)) + 0.5);
    int ho = float(std::max(std::abs(ho4 - ho1), std::abs(ho3 - ho2)) + 0.5);
    /* offset */
    float f1 = -0.5*(ho - 1)*cosTheta - 0.5*(wo - 1)*sinTheta + 0.5*(w - 1);
    float f2 =  0.5*(wo - 1)*sinTheta - 0.5*(ho - 1)*cosTheta + 0.5*(h - 1);
    xo = Tensor(ho, wo, c);
    for (int i = 0; i < ho; i++) {
        for(int j = 0; j < wo; j++) {
            int u = float(-j*sinTheta + i*cosTheta + f2 + 0.5);
            int v = float( j*cosTheta + i*sinTheta + f1 + 0.5);
            if (v >= 0 && v < w && u >= 0 && u < h) {
                for (int k = 0; k < c; k++) {
                    xo(ho - 1 - i, j, k) = xi(h - 1 - u, v, k);
                }
            }
       }
    }
    return 0;
}

int imp::nearestInterpolate(OutTensor xo, InTensor xi, const imp::Size &size)
{
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    int c = xi.shape[HWC_C];
    int ho = size.x;
    int wo = size.y;
    xo = Tensor(ho, wo, c);
    float rh = float(h)/float(ho);
    float rw = float(w)/float(wo);
    for (int i = 1; i < ho + 1; i++) {
        for (int j = 1; j < wo + 1; j++) {
            int u = imp::bound(i*rh + 0.5, 1, h + 1);
            int v = imp::bound(j*rw + 0.5, 1, w + 1);
            for (int k = 0; k < c; k++) {
                xo(i - 1, j - 1, k) = xi(u - 1, v - 1, k);
            }
        }
    }
    return 0;
}

int imp::bilinearInterpolate(OutTensor xo, InTensor xi, const Size &size)
{
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    int c = xi.shape[HWC_C];
    int ho = size.x;
    int wo = size.y;
    xo = Tensor(ho, wo, c);
    float rh = float(h)/float(ho);
    float rw = float(w)/float(wo);
    for (int i = 0; i < ho; i++) {
        for (int j = 0; j < wo; j++) {
            float si = i*rh;
            float sj = j*rw;
            float wi = si - int(si);
            float wj = sj - int(sj);
            int u0 = si;
            int u1 = u0 + 1;
            int v0 = sj;
            int v1 = v0 + 1;
            u1 = u0 + 1 >= h ? h - 1 : u0 + 1;
            v1 = v0 + 1 >= w ? w - 1 : v0 + 1;
            for (int k = 0; k < c; k++) {
                float y1 = xi(u0, v0, k)*(1 - wi) + xi(u1, v0, k)*wi;
                float y2 = xi(u0, v1, k)*(1 - wi) + xi(u1, v1, k)*wi;
                float y = y2*wj + (1 - wj)*y1;
                xo(i, j, k) = imp::bound(y, 0, 255);
            }
        }
    }
    return 0;
}


float imp::cubic::triangle(float x)
{
    float y = 0.5*x;
    return y < 0 ? y + 1 : 1 - y;
}

float imp::cubic::bell(float x)
{
    float y = x/2.0*1.5;
    float r = 0;
    if (y > -1.5 && y < -0.5 ) {
        r = 0.5 *(y + 1.5)*(y + 1.5);
    } else if (y > -0.5 && y < 0.5 ) {
        r = 0.75 - y*y;
    } else if (y > 0.5 && y < 1.5) {
        r = 0.5 * (y - 1.5)*(y - 1.5);
    }
    return r;
}

float imp::cubic::bspLine(float x)
{
    float xi = 0;
    float r = 1;
    if (x < 0) {
        xi = -x;
    }
    if (xi >= 0 && xi <= 1) {
        r = 2.0/3.0 + 0.5*xi*xi*xi - xi*xi;
    } else if (xi > 1 && xi <= 2 ){
        r = (2.0 - xi)*(2.0 - xi)*(2.0 - xi)/6;
    }
    return r;
}

int imp::cubicInterpolate(OutTensor xo, InTensor xi, const imp::Size &size, const std::function<float(float)> &interpolate)
{
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    int c = xi.shape[HWC_C];
    int ho = size.x;
    int wo = size.y;
    float rh = float(h)/float(ho);
    float rw = float(w)/float(wo);
    xo = Tensor(ho, wo, c);
    for (int i = 0; i < ho; i++) {
        for (int j = 0; j < wo; j++) {
            float si = i*rh;
            float sj = j*rw;
            for (int k = 0; k < c; k++) {
                float coeff = 0;
                for (int u = -1; u < 3; u++) {
                    for (int v = -1; v < 3; v++) {
                        int ui = u + si;
                        int vj = v + sj;
                        if (ui < 0 || vj < 0 || ui >= h || vj >= w) {
                            continue;
                        }
                        float wi = interpolate(u - si + int(si));
                        float wj = interpolate(sj - int(sj) - v);
                        float wij = wi*wj;
                        xo(i, j, k) += xi(ui, vj, k)*wij;
                        coeff += wij;
                    }
                }
                xo(i, j, k) /= coeff;
            }
        }
    }
    return 0;
}

int imp::affine(OutTensor xo, InTensor xi, InTensor op)
{
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    int c = xi.shape[HWC_C];
    xo = Tensor(h, w, c);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            Tensor p1({3, 1}, {float(i), float(j), 1});
            Tensor p2(3, 1);
            /* transform: p2 = op^T*p1 */
            Tensor::MM::kikj(p2, op, p1);
            int u = p2[0];
            int v = p2[1];
            if (u < 0 || v < 0 || u >= h || v >= w) {
                continue;
            }
            for (int k = 0; k < c; k++) {
                xo(u, v, k) = xi(i, j, k);
            }
        }
    }
    return 0;
}
