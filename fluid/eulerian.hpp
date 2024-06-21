#ifndef EULERIAN_HPP
#define EULERIAN_HPP
#include <functional>
#include <cmath>
#include <array>
#include "../basic/tensor.hpp"
#include "../basic/linalg.h"
#include "physics.hpp"

/*
    reference:
                http://www.matthiasmueller.info/tenMinutePhysics
*/
class Eulerian
{
public:
    enum Field {
        FIELD_U = 0,
        FIELD_V,
        FIELD_S
    };
public:
    std::size_t domainHeight;
    std::size_t domainWidth;
    /* density */
    double rho;
    /* grid spacing */
    double h;
    /* overrelaxtion: 1 < o < 2 */
    double overRelaxation;
    /* horizontal velocity field: (h, w) */
    Tensord u;
    Tensord un;
    /* vertical velocity field: (h, w) */
    Tensord v;
    Tensord vn;
    /* pressure: (h, w) */
    Tensord p;
    /* (h, w) */
    Tensord s;
    /* (h, w) */
    Tensord M;
    Tensord Mn;
public:
    explicit Eulerian(int height, int width, double density=1000)
        :domainWidth(width), domainHeight(height), rho(density)
    {
        /* grid spacing */
        h = 1;
        /* overrelaxtion: 1 < o < 2 */
        overRelaxation = 1.9;
        /* velocity field: (h, w) */
        u = Tensord(height, width);
        v = Tensord(height, width);
        un = Tensord(height, width);
        vn = Tensord(height, width);
        /* pressure field: (h, w) */
        p = Tensord(height, width);
        /* (h, w) */
        s = Tensord(height, width);
        /* (h, w) */
        M  = Tensord(height, width);
        Mn = Tensord(height, width);
    }

    void integrateGravity(double dt)
    {
        for (int i = 1; i < domainHeight - 1; i++) {
            for (int j = 1; j < domainWidth - 1; j++) {
                v(i, j) += -physics::g * dt;
            }
        }
        return;
    }

    void solveIncompressibility(std::size_t iterateTimes, double dt)
    {
        double cp = rho*h/dt;
        for (std::size_t it = 0; it <iterateTimes; it++) {
            for (std::size_t i = 1; i < domainHeight - 1; i++) {
                for (std::size_t j = 1; j < domainWidth - 1; j++) {
                    double sij = s(i, j);
                    if (sij == 0) {
                        continue;
                    }
                    double sx0 = s(i, j - 1);
                    double sx1 = s(i, j + 1);
                    double sy0 = s(i - 1, j);
                    double sy1 = s(i + 1, j);
                    double s_ = sx0 + sx1 + sy0 + sy1;
                    if (s_ == 0) {
                        continue;
                    }
                    double div = u(i, j + 1) - u(i, j) +  v(i + 1, j) - v(i, j);

                    double pij = -div/s_*overRelaxation*cp;
                    p(i, j) += pij;

                    u(i, j)     -= sx0*pij;
                    u(i, j + 1) += sx1*pij;
                    v(i, j)     -= sy0*pij;
                    v(i + 1, j) += sy1*pij;
                }
            }
        }
        return;
    }

    void extrapolate()
    {
        for (std::size_t i = 0; i < domainHeight; i++) {
            u(i, 0) = u(i, 1);
            u(i, domainWidth - 1, 0) = u(i, domainWidth - 2);
        }
        for (std::size_t j = 0; j < domainWidth; j++) {
            v(0, j) = v(1, j);
            v(domainHeight - 1, j) = v(domainHeight - 2, j);
        }
        return;
    }

    double sampleField(std::size_t x, std::size_t y, int field)
    {
        double h1 = 1.0 / h;
        double h2 = 0.5 * h;

        x = std::max(std::min(x, std::size_t(domainWidth * h)), std::size_t(h));
        y = std::max(std::min(y, std::size_t(domainHeight * h)), std::size_t(h));

        double dx = 0.0;
        double dy = 0.0;

        Tensord *pField = nullptr;

        switch (field) {
        case FIELD_U: pField = &u; dy = h2; break;
        case FIELD_V: pField = &v; dx = h2; break;
        case FIELD_S: pField = &M; dx = h2; dy = h2; break;
        }
        Tensord &f = *pField;

        std::size_t x0 = std::min(std::size_t(std::floor((x - dx)*h1)), domainWidth - 1);
        double tx = ((x-dx) - x0*h) * h1;
        std::size_t x1 = std::min(x0 + 1, domainWidth - 1);

        std::size_t y0 = std::min(std::size_t(std::floor((y - dy)*h1)), domainHeight - 1);
        double ty = ((y-dy) - y0*h) * h1;
        std::size_t y1 = std::min(y0 + 1, domainHeight - 1);

        std::size_t sx = 1.0 - tx;
        std::size_t sy = 1.0 - ty;

        double val = sx*sy * f(y0, x0) +
                     tx*sy * f(y0, x1) +
                     tx*ty * f(y1, x1) +
                     sx*ty * f(y1, x0);

        return val;
    }

    void advectVelocity(double dt)
    {
        un = u;
        vn = v;

        std::size_t h1 = h;
        std::size_t h2 = 0.5 * h;

        for (std::size_t i = 1; i < domainWidth; i++) {
            for (std::size_t j = 1; j < domainHeight; j++) {
                // u component
                if (s(i, j) != 0.0 && s(i-1, j) != 0.0 && j < domainHeight - 1) {
                    std::size_t x = i*h1;
                    std::size_t y = j*h1 + h2;
                    double uij = u(i, j);
                    double vij = (v(i-1, j) + v(i, j) + v(i-1, j+1) + v(i, j+1)) * 0.25;

                    x = x - dt*uij;
                    y = y - dt*vij;
                    uij = sampleField(x, y, FIELD_U);
                    un(i, j) = uij;
                }
                // v component
                if (s(i, j) != 0.0 && s(i, j-1) != 0.0 && i < domainWidth - 1) {
                    std::size_t x = i*h1 + h2;
                    std::size_t y = j*h1;
                    double uij = (u(i, j-1) + u(i, j) + u(i+1, + j-1) + u(i+1, j)) * 0.25;
                    double vij = v(i, j);
                    x   = x - dt*uij;
                    vij = y - dt*vij;
                    v = sampleField(x, y, FIELD_V);
                    vn(i, j) = vij;
                }
            }
        }
        u = un;
        v = vn;
        return;
    }

    void advectSmoke(double dt)
    {
        Mn = M;
        std::size_t h1 = h;
        std::size_t h2 = 0.5 * h;
        for (std::size_t i = 1; i < domainHeight-1; i++) {
            for (std::size_t j = 1; j < domainWidth-1; j++) {

                if (s(i, j) != 0.0) {
                    double uij = (u(i, j) + u(i+1, j)) * 0.5;
                    double vij = (v(i, j) + v(i, j+1)) * 0.5;
                    std::size_t x = i*h1 + h2 - dt*uij;
                    std::size_t y = j*h1 + h2 - dt*vij;
                    Mn(i, j) = sampleField(x, y, FIELD_S);
                }
            }
        }
        M = Mn;
        return;
    }

    void simulate(std::size_t iterateTimes, double dt)
    {
        /* gravity */
        integrateGravity(dt);
        /* solve incompressibility */
        p = 1;
        solveIncompressibility(iterateTimes, dt);
        /* extrapolate */
        extrapolate();
        /* advect velocity */
        advectVelocity(dt);
        /* advect smoke */
        advectSmoke(dt);
        return;
    }
};

#endif // EULERIAN_HPP
