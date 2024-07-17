#include "graphic3d.h"

Tensor imp::ball(int x0, int y0, int z0, float r, const Color3 &color)
{
    int d = 2*r;
    Tensor b(d, d, d, 3);
    for (int x = 0; x < d; x++) {
        for (int y = 0; y < d; y++) {
            for (int z = 0; z < d; z++) {
                float xd = x - x0;
                float yd = y - y0;
                float zd = z - z0;
                float radius = std::sqrt(xd*xd + yd*yd + zd*zd);
                if (radius == r) {
                    b(x, y, z, 0) = color.r;
                    b(x, y, z, 1) = color.g;
                    b(x, y, z, 2) = color.b;
                }
            }
        }
    }
    return b;
}

int imp::planeToSphere(OutTensor xo, InTensor xi, int r)
{

    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    int c = xi.shape[HWC_C];
    int d = 2*r;
    xo = Tensor(d, d, c);

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            /* plane to unitary ball */
            float u = 1.0 - 2.0*(i + 0.5)/h;
            float v = 2.0*(j + 0.5)/w - 1;
            float norm = std::sqrt(u*u + v*v);
            if (norm > 1) {
                continue;
            }
            /* sphere coordination */
            float phi = std::acos(u);
            float theta = std::atan2(u, v);
            int ur = r + r*std::sin(phi)*std::sin(theta);
            int vr = r + r*std::sin(phi)*std::cos(theta);
            //std::cout<<"theta="<<theta<<", phi="<<phi<<", u="<<ur<<", v="<<vr<<std::endl;
            if (ur >= 0 && vr >= 0 && ur < d && vr < d) {
                for (int k = 0; k < c; k++) {
                    xo(ur, vr, k) = xi(i, j, k);
                }
            }
        }
    }
    return 0;
}

