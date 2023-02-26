#ifndef MATS_HPP
#define MATS_HPP
#include <memory>
#include <vector>
#include "expt.hpp"
#include "exprfunc.hpp"

template<std::size_t ROW, std::size_t COL>
class Mats : public expt::Expr<Mats<ROW, COL>>
{
public:
    using T = expt::T;
    constexpr static std::size_t N = ROW*COL;
    constexpr static std::size_t rows = ROW;
    constexpr static std::size_t cols = COL;
public:
    std::vector<expt::T> val;
public:
    inline T operator[](std::size_t i) const {return val[i];}
    inline T& operator[](std::size_t i) {return val[i];}

    template<std::size_t Ni>
    inline expt::SubExpr<Mats, Mats<Ni, 1>> operator[](const Mats<Ni, 1> &indexes)
    {
        return expt::SubExpr<Mats, Mats<Ni, 1>>(*this, indexes);
    }

    template<std::size_t Ni>
    inline decltype(auto) operator[](const Mats<Ni, 1> &indexes) const
    {
        return expt::SubExpr<Mats, Mats<Ni, 1>>(*this, indexes);
    }

    inline T& at(std::size_t i) {return val[i];}
    inline std::size_t size() const {return val.size();}
    inline T operator()(std::size_t i, std::size_t j) const {return val[i*COL + j];}
    inline T& operator()(std::size_t i, std::size_t j) {return val[i*COL + j];}
    Mats():val(N, 0){}
    Mats(T value):val(N, value){}
    Mats(const std::initializer_list<T> &list)
        : val(N, 0)
    {
        val.assign(list.begin(), list.end());
    }
    ~Mats(){}
    Mats(const Mats &r):val(r.val){}

    template<typename Exprt>
    Mats(const expt::Expr<Exprt> &r)
    {
        const Exprt& expr = r.impl();
        val = std::vector<T>(expr.size());
        for (std::size_t i = 0; i < N; i++) {
            val[i] = expr[i];
        }
    }

    Mats& operator = (const Mats &r)
    {
        if (this == &r) {
            return *this;
        }
        val = r.val;
        return *this;
    }
    template<typename Exprt>
    Mats& operator = (const expt::Expr<Exprt> &r)
    {
        const Exprt& expr = r.impl();
        val = std::vector<T>(expr.size());
        for (std::size_t i = 0; i < N; i++) {
            val[i] = expr[i];
        }
        return *this;
    }

    Mats& operator += (const Mats &r)
    {
        expt::BinaryEvaluator<Mats, expt::Add, N>::impl(*this, *this, r);
        return *this;
    }

    Mats& operator -= (const Mats &r)
    {
        expt::BinaryEvaluator<Mats, expt::Sub, N>::impl(*this, *this, r);
        return *this;
    }

    Mats& operator *= (const Mats &r)
    {
        expt::BinaryEvaluator<Mats, expt::Mul, N>::impl(*this, *this, r);
        return *this;
    }

    Mats& operator /= (const Mats &r)
    {
        expt::BinaryEvaluator<Mats, expt::Div, N>::impl(*this, *this, r);
        return *this;
    }

    Mats& operator += (T x)
    {
        expt::BinaryEvaluator<Mats, expt::Add, N>::impl(*this, *this, x);
        return *this;
    }

    Mats& operator -= (T x)
    {
        expt::BinaryEvaluator<Mats, expt::Sub, N>::impl(*this, *this, x);
        return *this;
    }

    Mats& operator *= (T x)
    {
        expt::BinaryEvaluator<Mats, expt::Mul, N>::impl(*this, *this, x);
        return *this;
    }

    Mats& operator /= (T x)
    {
        expt::BinaryEvaluator<Mats, expt::Div, N>::impl(*this, *this, x);
        return *this;
    }

    Mats<COL, ROW> tr() const
    {
        Mats<COL, ROW> y;
        for (std::size_t i = 0; i < ROW; i++) {
            for (std::size_t j = 0; j < COL; j++) {
                y(j, i) = *this(i, j);
            }
        }
        return y;
    }


    void show() const
    {
        for (size_t i = 0; i < ROW; i++) {
            for (std::size_t j = 0; j < COL; j++) {
                std::cout<<val[i*COL + j]<<" ";
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
        return;
    }

};

template <std::size_t N>
using CVector = Mats<N, 1>;

template <std::size_t N>
using RVector = Mats<1, N>;


namespace MatsFunc {
    template<std::size_t I, std::size_t J, std::size_t K>
    Mats<I, K> mul(const Mats<I, J> &x1, const Mats<J, K> &x2)
    {
        Mats<I, K> y;
        for (std::size_t s = 0; s < y.size(); s++) {
            std::size_t i = s / y.cols;
            std::size_t j = s % y.cols;
            for (std::size_t k = 0; k < x1.cols; k++) {
                y.val[s] += x1.val[i*x1.cols + k] * x2.val[k*x2.cols + j];
            }
        }
        return y;
    }
}
#endif // MATS_HPP
