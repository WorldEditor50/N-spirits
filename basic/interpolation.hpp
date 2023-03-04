#ifndef INTERPOLATION_HPP
#define INTERPOLATION_HPP
#include <cmath>
#include <vector>

namespace Interpolate {

template<typename T>
class Lagrange
{
public:
    std::size_t n;
    std::vector<T> x;
    std::vector<T> y;
public:
    Lagrange():n(3){}
    explicit Lagrange(const std::vector<T> &x_, const std::vector<T> &y_, std::size_t n_)
        :n(n_),x(x_),y(y_){}
    T operator()(T x_)
    {
        /*
            (xi, yi): interpolation point
            li(x) = ∏（x - xi)/(xi - xj), i!=j
            Ln(x) = Σ li(x)*yi
        */
        T ln = 0;
        for (std::size_t i = 0; i < n; i++) {
            T li = 1;
            for (std::size_t j = 0; j < n; j++) {
                if (i != j) {
                    li *= (x_ - x[j])/(x[i] - x[j]);
                }
            }
            ln += li*y[i];
        }
        return ln;
    }
};


template<typename T>
class Newton
{
public:
    std::size_t n;
    std::vector<T> x;
    std::vector<T> y;
public:
    Newton():n(3){}
    explicit Newton(const std::vector<T> &x_, const std::vector<T> &y_, std::size_t n_)
        :n(n_),x(x_),y(y_){}
    T operator()(T x_)
    {
        T s = 0;
        std::vector<std::vector<T> > delta(x.size(), std::vector<T>(x.size(), 0));
        delta[0] = y;
        /* first order difference */
        for (std::size_t i = 1; i < x.size(); i++) {
            delta[1][i] = (delta[0][i] - delta[0][i - 1])/(x[i] - x[i - 1]);
        }
        /* high order difference */
        for (std::size_t i = 2; i < x.size(); i++) {
            for (std::size_t j = i; j < x.size(); j++) {
                delta[i][j] = (delta[i - 1][j] - delta[i - 1][j - 1])/(x[i] - x[0]);
            }
        }
        /* approximate */
        for (std::size_t i = 0; i < n; i++) {
            T p = 1;
            for (std::size_t j = 0; j < n; j++) {
                p *= x_ - x[j];
            }
            s += p*delta[i][i];
        }
        return s;
    }
};

}
#endif // INTERPOLATION_HPP
