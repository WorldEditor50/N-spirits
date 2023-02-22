#ifndef EXPRFUNC_HPP
#define EXPRFUNC_HPP
#include "expt.hpp"


namespace expt {

/* basic operator wrapper */
struct Plus { inline static T apply(T x1, T x2) {return x1 + x2;} };
struct Minus { inline static T apply(T x1, T x2) {return x1 - x2;} };
struct Multi { inline static T apply(T x1, T x2) {return x1 * x2;} };
struct Divide { inline static T apply(T x1, T x2) {return x1 / x2;} };
struct Negative { inline static T apply(T x) {return -x;} };
/* function wrapper */
struct Pow { inline static T apply(T x, T n) {return std::pow(x, n);} };
struct Sqrt { inline static T apply(T x) {return std::sqrt(x);} };
struct Exp { inline static T apply(T x) {return std::exp(x);} };
struct Tanh { inline static T apply(T x) {return std::tanh(x);} };
struct Sigmoid { inline static T apply(T x) {return 1 / (1 + std::exp(-x));} };
struct Relu { inline static T apply(T x) {return x > 0 ? x : 0;} };

/* expr function */
template<typename TExpr>
inline UnaryOp<Sqrt, TExpr>
sqrt(const Expr<TExpr> &expr)
{
    return UnaryOp<Sqrt, TExpr>(expr);
}

template<typename TExpr>
inline UnaryOp<Exp, TExpr>
exp(const Expr<TExpr> &expr)
{
    return UnaryOp<Exp, TExpr>(expr);
}

template<typename TExpr>
inline UnaryOp<Tanh, TExpr>
tanh(const Expr<TExpr> &expr)
{
    return UnaryOp<Tanh, TExpr>(expr);
}

template<typename TExpr>
inline UnaryOp<Sigmoid, TExpr>
sigmoid(const Expr<TExpr> &expr)
{
    return UnaryOp<Sigmoid, TExpr>(expr);
}

template<typename TExpr>
inline UnaryOp<Relu, TExpr>
relu(const Expr<TExpr> &expr)
{
    return UnaryOp<Relu, TExpr>(expr);
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

} // expt



/* global operator */
template<typename Left, typename Right>
inline expt::BinaryOp<expt::Plus, Left, Right>
operator + (const expt::Expr<Left> &left_, const expt::Expr<Right> &right_)
{
    return expt::BinaryOp<expt::Plus, Left, Right>(left_, right_);
}

template<typename Left, typename Right>
inline expt::BinaryOp<expt::Minus, Left, Right>
operator - (const expt::Expr<Left> &left_, const expt::Expr<Right> &right_)
{
    return expt::BinaryOp<expt::Minus, Left, Right>(left_, right_);
}

template<typename Left, typename Right>
inline expt::BinaryOp<expt::Multi, Left, Right>
operator * (const expt::Expr<Left> &left_, const expt::Expr<Right> &right_)
{
    return expt::BinaryOp<expt::Multi, Left, Right>(left_, right_);
}

template<typename Left, typename Right>
inline expt::BinaryOp<expt::Divide, Left, Right>
operator / (const expt::Expr<Left> &left_, const expt::Expr<Right> &right_)
{
    return expt::BinaryOp<expt::Divide, Left, Right>(left_, right_);
}

/* negative */
template<typename Right>
inline expt::UnaryOp<expt::Negative, Right>
operator - (const expt::Expr<Right> &right)
{
    return expt::UnaryOp<expt::Negative, Right>(right);
}

/* scalar operator */
template<typename Left>
inline expt::BinaryOp<expt::Plus, Left, expt::Scalar>
operator + (const expt::Expr<Left> &left_, expt::T right_)
{
    return expt::BinaryOp<expt::Plus, Left, expt::Scalar>(left_, Scalar(right_));
}

template<typename Left>
inline expt::BinaryOp<expt::Minus, Left, expt::Scalar>
operator - (const expt::Expr<Left> &left_, expt::T right_)
{
    return expt::BinaryOp<expt::Minus, Left, expt::Scalar>(left_, Scalar(right_));
}

template<typename Left>
inline expt::BinaryOp<expt::Multi, Left, expt::Scalar>
operator * (const expt::Expr<Left> &left_, expt::T right_)
{
    return expt::BinaryOp<expt::Multi, Left, expt::Scalar>(left_, Scalar(right_));
}

template<typename Left>
inline expt::BinaryOp<expt::Divide, Left, expt::Scalar>
operator / (const expt::Expr<Left> &left_, expt::T right_)
{
    return expt::BinaryOp<expt::Divide, Left, expt::Scalar>(left_, Scalar(right_));
}

/* scalar on the left hand side */
template<typename Right>
inline expt::BinaryOp<expt::Plus, expt::Scalar, Right>
operator + (expt::T left_, const expt::Expr<Right> & right_)
{
    return expt::BinaryOp<expt::Plus, expt::Scalar, Right>(Scalar(left_), right_);
}

template<typename Right>
inline expt::BinaryOp<expt::Minus, expt::Scalar, Right>
operator - (expt::T left_, const expt::Expr<Right> & right_)
{
    return expt::BinaryOp<expt::Minus, expt::Scalar, Right>(Scalar(left_), right_);
}

template<typename Right>
inline expt::BinaryOp<expt::Multi, expt::Scalar, Right>
operator * (expt::T left_, const expt::Expr<Right> & right_)
{
    return expt::BinaryOp<expt::Multi, expt::Scalar, Right>(Scalar(left_), right_);
}

#endif // EXPRFUNC_HPP
