#ifndef LBM_H
#define LBM_H
#include <functional>
#include <cmath>
#include <array>
#include "../basic/tensor.hpp"
#include "../basic/linalg.h"
#include "physics.hpp"

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
    explicit Cylinder(int ny, int nx, int r):x(nx),y(ny),radius(r){}
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

class Square
{
public:
    int x;
    int y;
    int r;
public:
    Square():r(0){}
    explicit Square(int ny, int nx, int r):x(nx),y(ny),r(r){}
    explicit Square(const Square &ref):x(ref.x),y(ref.y),r(ref.r){}
    bool isInside(int yi, int xi) const
    {
        int d = std::abs(xi - x) + std::abs(yi - y);
        if (d < r) {
            return true;
        }
        return false;
    }
};

class Cross
{
public:
    int x;
    int y;
    int r;
public:
    Cross():r(0){}
    explicit Cross(int ny, int nx, int r):x(nx),y(ny),r(r){}
    explicit Cross(const Cross &ref):x(ref.x),y(ref.y),r(ref.r){}
    bool isInside(int yi, int xi) const
    {
        int xt = xi - x;
        int yt = yi - y;
        if (std::abs(xt) + std::abs(yt) >= r) {
            return false;
        }
        int d  = 0;
        if (xt > 0 && yt < 0 ) {
            d = (xi - r)*(xi - r) + (yi - r)*(yi - r);
        }
        if (xt < 0 && yt < 0 ) {
            d = (xi + r)*(xi + r) + (yi - r)*(yi - r);
        }
        if (xt > 0 && yt > 0 ) {
            d = (xi + r)*(xi + r) + (yi + r)*(yi + r);
        }
        if (xt < 0 && yt > 0 ) {
            d = (xi - r)*(xi - r) + (yi + r)*(yi + r);
        }
        return d < r*r;
    }
};

class ICylinder
{
public:
    int x;
    int y;
    int r;
public:
    ICylinder():r(0){}
    explicit ICylinder(int ny, int nx, int r):x(nx),y(ny),r(r){}
    explicit ICylinder(const ICylinder &ref):x(ref.x),y(ref.y),r(ref.r){}
    bool isInside(int yi, int xi) const
    {
        int xt = xi - x;
        int yt = yi - y;
        if (std::abs(xt) + std::abs(yt) >= r) {
            return false;
        }
        int d  = 0;
        if (xt >= 0 && yt <= 0 ) {
            d = (xt - r)*(xt - r) + (yt + r)*(yt + r);
        }
        if (xt <= 0 && yt <= 0 ) {
            d = (xt + r)*(xt + r) + (yt + r)*(yt + r);
        }
        if (xt >= 0 && yt >= 0 ) {
            d = (xt - r)*(xt - r) + (yt - r)*(yt - r);
        }
        if (xt <= 0 && yt >= 0 ) {
            d = (xt + r)*(xt + r) + (yt - r)*(yt - r);
        }
        return d >= r*r;
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

#define LBM_MRT 1

/*

    Re = ρ*v*d/μ
    ρ: density
    v: velocity
    d: size of Obstacle
    μ: fluid viscosity

    laminar flow: Re < 2300
    turbulent flow: Re > 4000
*/
template <typename Object>
class LBM2d
{
public:
    enum Boundary {
        BOUNDARY_TOP = 0,
        BOUNDARY_RIGHT = 1,
        BOUNDARY_BOTTOM = 2,
        BOUNDARY_LEFT = 3
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
    double sigma;
    /* weights for velocity.: (9) */
    static Tensor w;
    /* lattice vector: (9, 2) */
    static Tensor e;
    /* density: (ny, nx) */
    Tensor rho;
    /* velocity: (ny, nx, 2) */
    Tensor vel;
    Tensor mask;
    /* particle density function: (ny, nx, grid.shape) */
    Tensor fn;
    Tensor f;
    /* boundary type: (top, right, bottom, left) */
    Tensor boundaryType;
    /* boundary value: (4, 2): (ny, nx)  */
    Tensor boundaryValue;
    /* Object */
    Object object;
public:
    LBM2d(){}
    LBM2d(int ny_, int nx_,
        const Object &obj,
        double niu_,
        const Tensor &boundaryType_,
        const Tensor &boundaryValue_)
        :nx(nx_),ny(ny_),niu(niu_),
          boundaryType(boundaryType_), boundaryValue(boundaryValue_),
          object(obj)
    {
        tau = 3*niu + 0.5;
        sigma = 1.0 / tau;
        /* density: (ny, nx) */
        rho = Tensor(ny, nx);
        rho.fill(1.0);
        /* velocity: (ny, nx, 2) */
        vel = Tensor(ny, nx, 2);
        mask = Tensor(ny, nx);
        /* particle density function: (ny, nx, grid.shape=9) */
        fn = Tensor(ny, nx, 9);
        f  = Tensor(ny, nx, 9);
        /* init */
        for (int i = 0; i < fn.shape[0]; i++) {
            for (int j = 0; j < fn.shape[1]; j++) {
                for (int k = 0; k < fn.shape[2]; k++) {
                    double value = feq(i, j, k);
                    fn(i, j, k) = value;
                    f(i, j, k) = value;
                }
                //f(i, j, 3) = 3;
            }
        }
        /* mask */
        for (int i = 0; i < mask.shape[0]; i++) {
            for (int j = 0; j < mask.shape[1]; j++) {
#if LBM_MRT
                /*
                   half way bounce back for no-slip boundary
                   borders
                */
                if (j == nx || i == 0 || i == ny) {
                     mask(i, j) = 1.0;
                }
#endif
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
                    fn(i, j, k) = (1 - sigma) * f(ip, jp, k) + feq(ip, jp, k) * sigma;
                }
            }
        }
        return;
    }

    static Tensor toMoment(const Tensor& f)
    {
        static Tensor M({9, 9}, {
                1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
               -4.0, -1.0, -1.0, -1.0, -1.0,  2.0,  2.0,  2.0,  2.0,
                4.0, -2.0, -2.0, -2.0, -2.0,  1.0,  1.0,  1.0,  1.0,
                0.0,  1.0,  0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0,
                0.0, -2.0,  0.0,  2.0,  0.0,  1.0, -1.0, -1.0,  1.0,
                0.0,  0.0,  1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0,
                0.0,  0.0, -2.0,  0.0,  2.0,  1.0,  1.0, -1.0, -1.0,
                0.0,  1.0, -1.0,  1.0, -1.0,  0.0,  0.0,  0.0,  0.0,
                0.0,  0.0,  0.0,  0.0,  0.0,  1.0, -1.0,  1.0, -1.0});
        Tensor r(9, 1);
        Tensor::MM::ikkj(r, M, f);
        return r;
    }

    static Tensor fromMoment(const Tensor &m)
    {
        static Tensor d({9, 1}, {1.0/9, 1.0/36, 1.0/36, 1.0/6,
                                  1.0/12, 1.0/6, 1.0/12, 1.0/4, 1.0/4});

        static Tensor M({9, 9}, {1, -4,  4,  0,  0,  0,  0,  0,  0,
                                  1, -1, -2,  1, -2,  0,  0,  1,  0,
                                  1, -1, -2,  0,  0,  1, -2, -1,  0,
                                  1, -1, -2, -1,  2,  0,  0,  1,  0,
                                  1, -1, -2,  0,  0, -1,  2, -1,  0,
                                  1,  2,  1,  1,  1,  1,  1,  0,  1,
                                  1,  2,  1, -1, -1,  1,  1,  0, -1,
                                  1,  2,  1, -1, -1, -1, -1,  0,  1,
                                  1,  2,  1,  1,  1, -1, -1,  0, -1});
        Tensor r(9, 1);
        /* r = Mx(d*m) */
        Tensor::MM::ikkj(r, M, d*m);
        return r;
    }

    void colliding()
    {
        static Tensor feqs(9, 1);
        for (int i = 1; i < ny - 1; i++) {
            for (int j = 1; j < nx - 1; j++) {
                for (int k = 0; k < e.shape[0]; k++) {
                    feqs[k] = feq(i, j, k);
                }
                Tensor fij = f.sub(i, j).reshape(9, 1);
                Tensor meq = toMoment(feqs);
                Tensor m = toMoment(fij);
                static Tensor s({9, 1}, {1.0, 1.63, 1.14, 1.0, 1.92, 0.0, 1.92, sigma, sigma});
                s.val[7] = sigma;
                s.val[8] = sigma;
                /* MRT */
                m += (meq - m) * s;
                /* BGK */
#if 0
                m += (meq - m) * sigma;
#endif
                Tensor mi = fromMoment(m);
                std::copy(mi.begin(), mi.end(), f.at(i, j));

            }
        }
        return;
    }

    void streaming()
    {
        /* inverse index for halfway bounce back */
        static int bi[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
        for (int i = 1; i < ny - 1; i++) {
            for (int j = 1; j < nx - 1; j++) {
                for (int k = 0; k < e.shape[0]; k++) {
                    int ip = i - e(k, 0);
                    int jp = j - e(k, 1);
                    if (mask(ip, jp) == 0) {
                        fn(i, j, k) = f(ip, jp, k);
                    } else {
                        fn(i, j, k) = f(i, j, bi[k]);
                    }
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
    void applyBoundaryCondition(int outer, int direct, int ibc, int jbc, int inb, int jnb)
    {
        if (outer == 1) {
            if (boundaryType[direct] == BOUNDARY_DIRICHLET) {
                vel(ibc, jbc, 0) = boundaryValue(direct, 0);
                vel(ibc, jbc, 1) = boundaryValue(direct, 1);
            } else if (boundaryType[direct] == BOUNDARY_NEUMANN) {
                vel(ibc, jbc, 0) = vel(inb, jnb, 0);
                vel(ibc, jbc, 1) = vel(inb, jnb, 1);
            }
        }
        rho(ibc, jbc) = rho(inb, jnb);
        for (int k = 0; k < e.shape[0]; k++) {
#if LBM_MRT
            f(ibc, jbc, k) = feq(ibc, jbc, k);
#else
            f(ibc, jbc, k) = f(inb, jnb, k) + feq(ibc, jbc, k) - feq(inb, jnb, k);
#endif
        }
        return;
    }

    void applyBoundaryCondition()
    {
        for (int j = 1; j < nx - 1; j++) {
            applyBoundaryCondition(1, BOUNDARY_TOP, 0, j, 1, j);
            applyBoundaryCondition(1, BOUNDARY_BOTTOM, ny - 1, j, ny - 2, j);
        }

        for (int i = 0; i < ny; i++) {
            applyBoundaryCondition(1, BOUNDARY_RIGHT, i, nx - 1, i, nx - 2);
            applyBoundaryCondition(1, BOUNDARY_LEFT, i, 0, i, 1);
        }
        /* boundary */
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
        for (std::size_t it = 0; it < iteratNum; it++) {
#if LBM_MRT
            colliding();
            streaming();
#else
            collideStream();
#endif
            update();
            applyBoundaryCondition();
            Tensor img(ny, nx, 3);
            for (int i = 0; i < ny; i++) {
                for (int j = 0; j < nx; j++) {
                    double u = vel(i, j, 0);
                    double v = vel(i, j, 1);
                    double p = std::sqrt(u*u + v*v)*1e4;
                    img(i, j, 0) = colorScaler[0]*p;
                    img(i, j, 1) = colorScaler[1]*p;
                    img(i, j, 2) = colorScaler[2]*p;
                }
            }
            float c = img.max()/255.0;
            img /= c;
            func(it, img);
        }
        return;
    }

};
template <typename T>
Tensor LBM2d<T>::w({9}, {4.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
                      1.0 / 9.0,  1.0 / 9.0,  1.0 / 36.0,
                      1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0});
template <typename T>
Tensor LBM2d<T>::e({9, 2}, {0, 0,  1,  0, 0, 1,
                        -1, 0,  0, -1, 1, 1,
                        -1, 1, -1, -1, 1, -1});
#endif // LBM_H

