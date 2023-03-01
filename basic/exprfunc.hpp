#ifndef EXPRFUNC_HPP
#define EXPRFUNC_HPP
#include "expt.hpp"


namespace expt {

/* basic operator wrapper */
struct Add { inline static T apply(T x1, T x2) {return x1 + x2;} };
struct Sub { inline static T apply(T x1, T x2) {return x1 - x2;} };
struct Mul { inline static T apply(T x1, T x2) {return x1 * x2;} };
struct Div { inline static T apply(T x1, T x2) {return x1 / x2;} };
struct Neg { inline static T apply(T x) {return -x;} };
/* function wrapper */
struct Pow { inline static T apply(T x, T n) {return std::pow(x, n);} };
struct Sqrt { inline static T apply(T x) {return std::sqrt(x);} };
struct Exp { inline static T apply(T x) {return std::exp(x);} };
struct Tanh { inline static T apply(T x) {return std::tanh(x);} };
struct Sigmoid { inline static T apply(T x) {return 1 / (1 + std::exp(-x));} };
struct Relu { inline static T apply(T x) {return x > 0 ? x : 0;} };

/* expr function */
template<typename Exprt>
inline UnaryOp<Sqrt, Exprt>
sqrt(const Expr<Exprt> &expr)
{
    return UnaryOp<Sqrt, Exprt>(expr);
}

template<typename Exprt>
inline UnaryOp<Exp, Exprt>
exp(const Expr<Exprt> &expr)
{
    return UnaryOp<Exp, Exprt>(expr);
}

template<typename Exprt>
inline UnaryOp<Tanh, Exprt>
tanh(const Expr<Exprt> &expr)
{
    return UnaryOp<Tanh, Exprt>(expr);
}

template<typename Exprt>
inline UnaryOp<Sigmoid, Exprt>
sigmoid(const Expr<Exprt> &expr)
{
    return UnaryOp<Sigmoid, Exprt>(expr);
}

template<typename Exprt>
inline UnaryOp<Relu, Exprt>
relu(const Expr<Exprt> &expr)
{
    return UnaryOp<Relu, Exprt>(expr);
}


/* evaluate */
template<typename Type, typename Operator, std::size_t N>
struct BinaryEvaluator {
    inline static void impl(Type &x, const Type &x1, const Type &x2)
    {
        x[N - 1] = Operator::apply(x1[N - 1], x2[N - 1]);
        BinaryEvaluator<Type, Operator, N - 1>::impl(x, x1, x2);
        return;
    }
    inline static void impl(Type &x, const Type &x1, const T &x2)
    {
        x[N - 1] = Operator::apply(x1[N - 1], x2);
        BinaryEvaluator<Type, Operator, N - 1>::impl(x, x1, x2);
        return;
    }
};
template<typename Type, typename Operator>
struct BinaryEvaluator<Type, Operator, 0> {
    inline static void impl(Type &x, const Type &x1, const Type &x2)
    {
        x[0] = Operator::apply(x1[0], x2[0]);
        return;
    }
    inline static void impl(Type &x, const Type &x1, const T &x2)
    {
        x[0] = Operator::apply(x1[0], x2);
        return;
    }
};

template<typename Type, typename Operator, std::size_t N>
struct UnaryEvaluator {
    inline static void impl(Type &y, const Type &x)
    {
        y[N - 1] = Operator::apply(x[N - 1]);
        UnaryEvaluator<Type, Operator, N - 1>::impl(x);
        return;
    }
};
template<typename Type, typename Operator>
struct UnaryEvaluator<Type, Operator, 0> {
    inline static void impl(Type &y, const Type &x)
    {
        y[0] = Operator::apply(x[0]);
        return;
    }
};
/* dot product */
template<typename Type, std::size_t N>
struct Dot {
    inline static T eval(const Type &x1, const Type &x2)
    {
        return x1[N] * x2[N] + Dot<Type, N - 1>::eval(x1, x2);
    }
};
template<typename Type>
struct Dot<Type, 0> {
    inline static T eval(const Type &x1, const Type &x2){return x1[0] * x2[0];}
};
template <typename Type>
inline T dot(const Type &x1, const Type &x2)
{
    return Dot<Type, Type::N - 1>::eval(x1, x2);
}

/* sum */
template<typename Type, std::size_t N>
struct Sum {
    inline static T eval(const Type &x)
    {
        return x[N] + Dot<Type, N - 1>::eval(x);
    }
};
template<typename Type>
struct Sum<Type, 0> {
    inline static T eval(const Type &x){return x[0];}
};
template <typename Type>
inline T sum(const Type &x)
{
    return Dot<Type, Type::N - 1>::eval(x);
}



} // expt



/* global operator */
template<typename Left, typename Right>
inline expt::BinaryOp<expt::Add, Left, Right>
operator + (const expt::Expr<Left> &left_, const expt::Expr<Right> &right_)
{
    return expt::BinaryOp<expt::Add, Left, Right>(left_, right_);
}

template<typename Left, typename Right>
inline expt::BinaryOp<expt::Sub, Left, Right>
operator - (const expt::Expr<Left> &left_, const expt::Expr<Right> &right_)
{
    return expt::BinaryOp<expt::Sub, Left, Right>(left_, right_);
}

template<typename Left, typename Right>
inline expt::BinaryOp<expt::Mul, Left, Right>
operator * (const expt::Expr<Left> &left_, const expt::Expr<Right> &right_)
{
    return expt::BinaryOp<expt::Mul, Left, Right>(left_, right_);
}

template<typename Left, typename Right>
inline expt::BinaryOp<expt::Div, Left, Right>
operator / (const expt::Expr<Left> &left_, const expt::Expr<Right> &right_)
{
    return expt::BinaryOp<expt::Div, Left, Right>(left_, right_);
}

/* negative */
template<typename Right>
inline expt::UnaryOp<expt::Neg, Right>
operator - (const expt::Expr<Right> &right)
{
    return expt::UnaryOp<expt::Neg, Right>(right);
}

/* scalar operator */
template<typename Left>
inline expt::BinaryOp<expt::Add, Left, expt::Scalar>
operator + (const expt::Expr<Left> &left_, expt::T right_)
{
    return expt::BinaryOp<expt::Add, Left, expt::Scalar>(left_, expt::Scalar(right_));
}

template<typename Left>
inline expt::BinaryOp<expt::Sub, Left, expt::Scalar>
operator - (const expt::Expr<Left> &left_, expt::T right_)
{
    return expt::BinaryOp<expt::Sub, Left, expt::Scalar>(left_, expt::Scalar(right_));
}

template<typename Left>
inline expt::BinaryOp<expt::Mul, Left, expt::Scalar>
operator * (const expt::Expr<Left> &left_, expt::T right_)
{
    return expt::BinaryOp<expt::Mul, Left, expt::Scalar>(left_, expt::Scalar(right_));
}

template<typename Left>
inline expt::BinaryOp<expt::Div, Left, expt::Scalar>
operator / (const expt::Expr<Left> &left_, expt::T right_)
{
    return expt::BinaryOp<expt::Div, Left, expt::Scalar>(left_, expt::Scalar(right_));
}

/* scalar on the left hand side */
template<typename Right>
inline expt::BinaryOp<expt::Add, expt::Scalar, Right>
operator + (expt::T left_, const expt::Expr<Right> & right_)
{
    return expt::BinaryOp<expt::Add, expt::Scalar, Right>(expt::Scalar(left_), right_);
}

template<typename Right>
inline expt::BinaryOp<expt::Sub, expt::Scalar, Right>
operator - (expt::T left_, const expt::Expr<Right> & right_)
{
    return expt::BinaryOp<expt::Sub, expt::Scalar, Right>(expt::Scalar(left_), right_);
}

template<typename Right>
inline expt::BinaryOp<expt::Mul, expt::Scalar, Right>
operator * (expt::T left_, const expt::Expr<Right> & right_)
{
    return expt::BinaryOp<expt::Mul, expt::Scalar, Right>(expt::Scalar(left_), right_);
}

#endif // EXPRFUNC_HPP
