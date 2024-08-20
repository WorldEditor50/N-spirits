#ifndef OPTIMIZATION_HPP
#define OPTIMIZATION_HPP
#include <iostream>
#include <cmath>
#include <random>
#include "linalg.h"
#include "tensor.hpp"

namespace LinAlg {

class Wolfe
{
private:
    float rho;
    float sigma;
    float a;
    float b;
    int iterateTimes;
public:
    Wolfe():rho(0.4),sigma(0.5),a(0),b(MAXFLOAT),iterateTimes(100){}
    explicit Wolfe(float rho_, float sigma_, int iterateTimes_)
        :rho(rho_),sigma(sigma_), iterateTimes(iterateTimes_){}
    template<typename Fn>
    float operator()(Fn &fn,
                     const Tensor &x,
                     const Tensor &g,
                     const Tensor &d)
    {
        float alpha = 1;
        for (int i = 0; i < iterateTimes; i++) {
            Tensor delta = fn(x + d*alpha) - fn(x);
            if (delta.norm2() < rho*alpha*LinAlg::dot(fn.df(x), d)) {
                b = alpha;
                alpha = (a + alpha)/2;
                continue;
            }
            if (LinAlg::dot(fn.df(x + d*alpha), d) <
                    sigma*LinAlg::dot(fn.df(x), d)) {
                a = alpha;
                alpha = std::min(2*alpha, (alpha + b)/2);
                continue;
            }
            break;
        }
        return alpha;
    }

    template<typename Fn>
    float operator()(Fn &fn,
                     const Tensor &w,
                     const Tensor &x,
                     const Tensor &g,
                     const Tensor &d)
    {
        float alpha = 1;
        while (1) {
            Tensor delta = fn(w + d*alpha, x) - fn(w, x);
            if (delta.norm2() < (-rho*alpha*LinAlg::dot(fn.df(w, x), d))) {
                b = alpha;
                alpha = (a + alpha)/2;
                continue;
            }
            if (LinAlg::dot(fn.df(w + d*alpha, x), d) <
                    sigma * LinAlg::dot(fn.df(w, x), d)) {
                a = alpha;
                alpha = std::min(2*alpha, (alpha + b)/2);
                continue;
            }
            break;
        }
        return alpha;
    }
};

class Armijo
{
private:
    float rho;
    float sigma;
    int iterateTimes;
public:
    Armijo():rho(0.6),sigma(0.4),iterateTimes(200){}
    explicit Armijo(float rho_, float sigma_, int iterateTimes_)
        :rho(rho_),sigma(sigma_),iterateTimes(iterateTimes_){}
    template<typename Fn>
    float operator()(Fn &fn,
                     const Tensor &x,
                     const Tensor &g,
                     const Tensor &d)
    {
        float alpha = 1e-3;
        float r = rho;
        for (int i = 0; i < iterateTimes; i++) {
            Tensor delta = fn(x + d*r) - fn(x);
            float a = delta.norm2() - sigma*r*LinAlg::dot(g, d);
            if (a < 0) {
                alpha = r;
                std::cout<<"i="<<i<<",a="<<a<<",alpha="<<alpha<<std::endl;
                break;
            }
            r *= rho;
        }
        return alpha;
    }

    template<typename Fn>
    float operator()(Fn &fn,
                     const Tensor &w,
                     const Tensor &x,
                     const Tensor &g,
                     const Tensor &d)
    {
        float alpha = 1e-3;
        float r = rho;
        for (int i = 0; i < iterateTimes; i++) {
            Tensor delta = fn(w + d*r, x) - fn(w, x);
            if (delta.norm2() < sigma*r*LinAlg::dot(g, d)) {
                alpha = r;
                break;
            }
            r *= rho;
        }
        return alpha;
    }
};

class GradientDescent
{
private:
    int maxEpoch;
    float lr;
    float eps;
public:
    GradientDescent(){}
    explicit GradientDescent(int maxEpoch_, float lr_=1e-2, float eps_=1e-3)
        :maxEpoch(maxEpoch_),lr(lr_),eps(eps_){}
    template<typename Fn>
    int operator()(Fn &fn,
                   Tensor &w,
                   const std::vector<Tensor> &x,
                   const Tensor &yt)
    {
        /*
           x:(b, N)
           y:(b, 1) */
        for (int epoch = 0; epoch < maxEpoch; epoch++) {
            //std::size_t N = x[0].totalSize;
            std::size_t M = w.totalSize;
            /* error */
            Tensor y(x.size(), 1);
            Tensor e(x.size(), 1);
            for (std::size_t i = 0; i < x.size(); i++) {
                y[i] = fn(x[i]);
                e = y[i] - yt[i];
            }
            Tensor g(M, 1);
            for (std::size_t i = 0; i < x.size(); i++) {
                g += fn.df(x[i])*e[i];
            }
            for (std::size_t i = 0; i < M; i++) {
                w[i] -= lr*g[i];
            }
            if (e.norm2() < eps && epoch > maxEpoch/2) {
                break;
            }
        }
        return 0;
    }
};

template<typename LineSearch=Armijo>
class ConjugateGradient
{
private:
    int maxEpoch;
    float eps;
    LineSearch lineSearch;
public:
    ConjugateGradient(){}
    explicit ConjugateGradient(const LineSearch &lineSearch_, int maxEpoch_, float eps_=1e-3)
        :maxEpoch(maxEpoch_),eps(eps_),lineSearch(lineSearch_){}
    template<typename Fn>
    Tensor operator()(Fn &fn, const Tensor &x0)
    {
        Tensor x = x0;
        std::size_t N = x.totalSize;
        Tensor d;
        Tensor d0;
        Tensor g0;
        for (int epoch = 0; epoch < maxEpoch; epoch++) {
            Tensor g = fn.df(x);
            int itern = epoch % N;
            if (itern == 0) {
                d = -g;
            } else {
                float beta = LinAlg::dot(g, g)/(LinAlg::dot(g0, g0) + 1e-8);
                //std::cout<<"beta="<<beta<<std::endl;
                d = -g + d0*beta;
                float s = LinAlg::dot(g, d);
                if (s >= 0) {
                    d = -g;
                }
            }
            if (g.norm2() < eps) {
                std::cout<<"epoch="<<epoch<<",finished."<<std::endl;
                break;
            }
            float alpha = lineSearch(fn, x, g, d);
            x += d*alpha;
            g0 = g;
            d0 = d;
        }
        return x;
    }
};

template<typename LineSearch=Armijo>
class BFGS
{
private:
    int maxEpoch;
    float eps;
    LineSearch lineSearch;
public:
    BFGS(){}
    explicit BFGS(const LineSearch &lineSearch_, int maxEpoch_, float eps_=1e-3)
        :maxEpoch(maxEpoch_),eps(eps_),lineSearch(lineSearch_){}
    template<typename Fn>
    Tensor solve(Fn &fn, const Tensor &x0)
    {
        Tensor x = x0;
        int N = x.totalSize;
        Tensor H = LinAlg::eye(N);
        for (int epoch = 0; epoch < maxEpoch; epoch++) {
             Tensor g = fn.df(x);
             if (g.norm2() < eps) {
                 std::cout<<"epoch="<<epoch<<",finished."<<std::endl;
                 break;
             }
             Tensor d = -H%g;
             float alpha = lineSearch(fn, x, g, d);
             Tensor xd = d*alpha;
             x += xd;
             Tensor gd = fn.df(x) - g;
             if (LinAlg::dot(xd, gd) > 0) {
                 Tensor delta = xd - H%gd;
                 H += delta%delta.tr()/LinAlg::dot(delta, gd);
             }
        }
        return x;
    }

    template<typename Fn>
    int operator()(Fn &fn,
                   Tensor &w,
                   const std::vector<Tensor> &x)
    {
        int N = w.totalSize;
        Tensor H = LinAlg::eye(N);
        for (int epoch = 0; epoch < maxEpoch; epoch++) {
            for (std::size_t i = 0; i < x.size(); i++) {
                Tensor g = fn.df(w, x[i]);
                if (g.norm2() < eps) {
                    std::cout<<"epoch="<<epoch<<",finished."<<std::endl;
                    break;
                }
                Tensor d = -H%g;
                float alpha = lineSearch(fn, w, x[i], g, d);
                Tensor wd = d*alpha;
                w += wd;
                Tensor gn = fn.df(w, x[i]);
                Tensor gd = gn - g;
                if (LinAlg::dot(wd, gd) > 0) {
                    Tensor delta = wd - H%gd;
                    H += delta%delta.tr()/LinAlg::dot(delta, gd);
                }
            }
        }
        return 0;
    }
};

template<typename LineSearch=Armijo>
class DFP
{
private:
    int maxEpoch;
    float eps;
    LineSearch lineSearch;
public:
    DFP(){}
    explicit DFP(int maxEpoch_, float eps_=1e-3)
        :maxEpoch(maxEpoch_),eps(eps_){}
    template<typename Fn>
    Tensor operator()(Fn &fn, const Tensor &x0)
    {
        Tensor x = x0;
        int N = x.totalSize;
        Tensor H = LinAlg::eye(N);
        for (int epoch = 0; epoch < maxEpoch; epoch++) {
            Tensor g = fn.df(x);
            if (g.norm2() < eps) {
                break;
            }
            Tensor d = -H%g;
            float alpha = lineSearch(fn, x, g, d);
            Tensor xd = d*alpha;
            x += xd;
            Tensor gd = fn.df(x) - g;
            if (LinAlg::dot(xd, gd) > 0) {
               H += -(H%gd%gd.tr()%H)/(gd.tr()%H%gd) + (xd%xd.tr())/(xd.tr()%gd);
            }
        }
        return x;
    }
};

}

#endif // OPTIMIZATION_HPP
