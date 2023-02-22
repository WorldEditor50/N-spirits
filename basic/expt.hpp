#ifndef EXPT_HPP
#define EXPT_HPP
#include <iostream>
#include <cmath>
#include <type_traits>

/* expression template */
namespace expt {

using T = float;

template <typename ExprImpl>
class Expr
{
public:
    using Type = ExprImpl;
public:
    inline const ExprImpl& impl() const {return static_cast<const ExprImpl&>(*this);}
    inline std::size_t size() const {return static_cast<const ExprImpl&>(*this).size();}
};

/* scalar */
class Scalar : public Expr<Scalar>
{
protected:
    T s;
public:
    Scalar(const T &s_):s(s_){}
    Scalar(const Scalar &r):s(r.s){}
    inline T operator[](std::size_t) const {return s;}
    inline T &operator[](std::size_t) {return s;}
    inline std::size_t size() const {return 1;}
};

/* trait */
template<typename Exprt>
struct TraitRef {
    using type = typename std::conditional<std::is_same<Scalar, Exprt>::value,
                                           Scalar,
                                           const Exprt&>::type;
};

/* binary operator template */
template<typename Operator, typename Left, typename Right>
class BinaryOp : public Expr<BinaryOp<Operator, Left, Right> >
{
public:
    explicit BinaryOp(const Expr<Left> &left_, const Expr<Right> &right_):
        left(left_.impl()), right(right_.impl()){}
    inline T operator[](std::size_t i) const {return Operator::apply(left[i], right[i]);}
    inline T &operator[](std::size_t i) {return Operator::apply(left[i], right[i]);}
    inline size_t size() const {return left.size();}
protected:
    typename TraitRef<Left>::type left;
    typename TraitRef<Right>::type right;
};
/*  unary operator template */
template<typename Operator, typename Right>
class UnaryOp : public Expr<UnaryOp<Operator, Right> >
{
public:
    explicit UnaryOp(const Expr<Right> &right_):right(right_.impl()){}
    inline T operator[](std::size_t i) const {return Operator::apply(right[i]);}
    inline T &operator[](std::size_t i) {return Operator::apply(right[i]);}
    inline std::size_t size() const {return right.size();}
protected:
    typename TraitRef<Right>::type right;
};

}




#endif // EXPT_HPP
