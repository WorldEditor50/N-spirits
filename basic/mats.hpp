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
protected:
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

    template<std::size_t COL_>
    static Mats<ROW, COL_> mul(const Mats &x1, const Mats<COL, COL_> &x2)
    {
        Mats<ROW, COL_> y;
        for (std::size_t i = 0; i < y.rows; i++) {
            for (std::size_t j = 0; j < y.cols; j++) {
                for (std::size_t k = 0; k < cols; k++) {
                    y.val[i*y.cols + j] += x1.val[i*cols + k] * x2.val[k*x2.cols + j];
                }
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
using ColVector = Mats<N, 1>;

template <std::size_t N>
using RowVector = Mats<1, N>;

#endif // MATS_HPP
