#ifndef LBM_H
#define LBM_H
#include <functional>
#include <array>
#include "../basic/tensor.hpp"
#include "../basic/utils.h"

/*
        reference:
                    http://yangwc.com/2020/06/24/LBM/
                    https://forum.taichi-lang.cn/t/homework0/506

*/
class Cylinder
{
public:
    int x;
    int y;
    int radius;
public:
    Cylinder():x(0),y(0),radius(0){}
    explicit Cylinder(int ny, int nx, int r):x(nx/2),y(ny/4),radius(r){}
    explicit Cylinder(const Cylinder &ref):x(ref.x),y(ref.y),radius(ref.radius){}
    bool isInside(int yi, int xi) const
    {
        int d = (x - xi)*(x - xi) + (y - yi)*(y - yi);
        if (d < radius*radius) {
            return true;
        }
        return false;
    }
};

/*
    axis:
            o------------------> x
            |
            |   O cylinder
            |
            v y
*/
template <typename Object>
class LBM2d
{
public:
    enum Boundary {
        BOUNDARY_LEFT = 0,
        BOUNDARY_RIGHT,
        BOUNDARY_TOP,
        BOUNDARY_BUTTOM
    };
    enum BoundaryValue {
        BOUNDARY_DIRICHLET = 0,
        BOUNDARY_NEUMANN
    };
public:
    int nx;
    int ny;
    /* viscosity of fluid */
    double niu;
    double tau;
    /* density: (ny, nx) */
    Tensord rho;
    /* velocity: (ny, nx, 2) */
    Tensord vel;
    Tensor mask;
    /* particle density function: (ny, nx, grid.shape) */
    Tensord fn;
    Tensord f;
    /* weights for velocity.: (9) */
    Tensord w;
    /* lattice vector: (9, 2) */
    Tensord e;
    /* boundary type: (4) */
    Tensord boundaryType;
    /* boundary value: (4, 2) */
    Tensord boundaryValue;
    /* Object */
    Object object;
public:
    LBM2d(){}
    LBM2d(int nx_, int ny_,
        const Object &obj,
        double niu_,
        const Tensord &boundaryType_,
        const Tensord &boundaryValue_)
        :nx(nx_),ny(ny_),niu(niu_), boundaryType(boundaryType_), boundaryValue(boundaryValue_)
    {
        tau = 3*niu + 0.5;
        /* density: (ny, nx) */
        rho = Tensord(ny, nx);
        rho.fill(1.0);
        /* velocity: (ny, nx, 2) */
        vel = Tensord(ny, nx, 2);
        mask = Tensor(ny, nx);
        /* particle density function: (ny, nx, grid.shape=9) */
        fn = Tensord(ny, nx, 9);
        f  = Tensord(ny, nx, 9);
        /* weights for velocity.: (9) */
        w = Tensord({9}, {4.0/9.0, 1.0/9.0, 1.0/ 9.0,
                          1.0/9.0, 1.0/9.0, 1.0/36.0,
                          1.0/36.0, 1.0/36.0, 1.0/36.0});
        /* lattice vector: (9, 2) */
        e = Tensord({9, 2},{  0, 0,
                              1, 0,
                              0, 1,
                             -1, 0,
                              0,-1,
                              1, 1,
                             -1, 1,
                             -1,-1,
                              1,-1
                    });
        /* cylinder */
        object = Cylinder(ny, nx, 12);

        /* init */
        for (int i = 0; i < fn.shape[0]; i++) {
            for (int j = 0; j < fn.shape[1]; j++) {
                for (int k = 0; k < fn.shape[2]; k++) {
                    double value = feq(i, j, k);
                    fn(i, j, k) = value;
                    f(i, j, k) = value;
                }
            }
        }
        /* locate cylinder */
        for (int i = 0; i < mask.shape[0]; i++) {
            for (int j = 0; j < mask.shape[1]; j++) {
                if (object.isInside(i, j) == true) {
                    mask(i, j) = 1;
                } else {
                    mask(i, j) = 0;
                }
            }
        }

    }
    double feq(int i, int j, int k)
    {
        double u = vel(i, j, 0);
        double v = vel(i, j, 1);
        double eu = e(k, 0) * u + e(k, 1) * v;
        double uv = u*u + v*v;
        return w[k] * rho(i, j) * (1.0 + 3.0 * eu + 4.5 * eu*eu - 1.5 * uv);
    }

    void collideStream()
    {
        for (int i = 1; i < ny - 1; i++) {
            for (int j = 1; j < nx - 1; j++) {
                for (int k = 0; k < e.shape[0]; k++) {
                    int ip = i - e(k, 0);
                    int jp = j - e(k, 1);
                    fn(i, j, k) = (1 - 1.0/tau) * f(ip, jp, k) + feq(ip, jp, k) / tau;
                }
            }
        }
        return;
    }

    void update()
    {
        /* update f */
        f = fn;
        for (int i = 1; i < ny - 1; i++) {
            for (int j = 1; j < nx - 1; j++) {
                double r = 0;
                double u = 0;
                double v = 0;
                for (int k = 0; k < e.shape[0]; k++) {
                    /* calculate density */
                    double Fijk = fn(i, j, k);
                    r += Fijk;
                    /* velocity */
                    u += e(k, 0) * Fijk;
                    v += e(k, 1) * Fijk;
                }
                rho(i, j) = r;
                vel(i, j, 0) = u / r;
                vel(i, j, 1) = v / r;
            }
        }
        return;
    }
    void applyBoundaryCondition(int outer, int dr, int ibc, int jbc, int inb, int jnb)
    {
        if (outer == 1) {
            if (boundaryType[dr] == BOUNDARY_DIRICHLET) {
                vel(ibc, jbc, 0) = boundaryValue(dr, 0);
                vel(ibc, jbc, 1) = boundaryValue(dr, 1);
            } else if (boundaryType[dr] == BOUNDARY_NEUMANN) {
                vel(ibc, jbc, 0) = vel(inb, jnb, 0);
                vel(ibc, jbc, 1) = vel(inb, jnb, 1);
            }
        }
        rho(ibc, jbc) = rho(inb, jnb);
        for (int k = 0; k < e.shape[0]; k++) {
            f(ibc, jbc, k) = f(inb, jnb, k) + feq(ibc, jbc, k) - feq(inb, jnb, k);
        }
        return;
    }

    void applyBoundaryCondition()
    {
        for (int j = 1; j < nx - 1; j++) {
            /* left: dr = 0; ibc = 0; jbc = j; inb = 1; jnb = j */
            applyBoundaryCondition(1, 0, 0, j, 1, j);
            /* right: dr = 2; ibc = ny-1; jbc = j; inb = ny-2; jnb = j */
            applyBoundaryCondition(1, 2, ny - 1, j, ny - 2, j);
        }

        for (int i = 0; i < ny; i++) {
            /* top: dr = 1; ibc = i; jbc = nx-1; inb = i; jnb = nx-2 */
            applyBoundaryCondition(1, 1, i, nx - 1, i, nx - 2);
            /* bottom: dr = 3; ibc = i; jbc = 0; inb = i; jnb = 1 */
            applyBoundaryCondition(1, 3, i, 0, i, 1);
        }
        /* cylinder boundary */
        for (int i = 1; i < ny - 1; i++) {
            for (int j = 1; j < nx - 1; j++) {
                if (mask(i, j) == 0) {
                    continue;
                }
                vel(i, j, 0) = 0;
                vel(i, j, 1) = 0;
                int inb = 0;
                int jnb = 0;
                if (i >= object.y) {
                    inb = i + 1;
                } else {
                    inb = i - 1;
                }
                if (j >= object.x) {
                    jnb = j + 1;
                } else {
                    jnb = j - 1;
                }
                applyBoundaryCondition(0, 0, i, j, inb, jnb);
            }
        }
        return;
    }

    void solve(std::size_t iteratNum, const std::array<double, 3> &colorScaler, std::function<void(std::size_t i, Tensor &img)> func)
    {
        for (std::size_t i = 0; i < iteratNum; i++) {
            collideStream();
            update();
            applyBoundaryCondition();
            Tensor img(ny, nx, 3);
            for (int i = 0; i < ny; i++) {
                for (int j = 0; j < nx; j++) {
                    double u = vel(i, j, 0);
                    double v = vel(i, j, 1);
                    double p = std::sqrt(u*u + v*v);
                    img(i, j, 0) = colorScaler[0] * 1000*p;
                    img(i, j, 1) = colorScaler[1] * 1000*p;
                    img(i, j, 2) = colorScaler[2] * 1000*p;
                }
            }
            float c = img.max()/255.0;
            img /= c;
            func(i, img);
        }
        return;
    }

};
#endif // LBM_H

