#ifndef TENSORSI_HPP
#define TENSORSI_HPP
#include "tensor.hpp"
#include "sse2func.hpp"
#include "avx2func.hpp"

template<typename T , typename Instruct, template<typename Ti> class Alloc=AlignAllocator32>
class Tensorsi_ : public Tensor_<T, Alloc>
{
public:
    using __Tensor = Tensor_<T, Alloc>;
    using __Tensor::totalSize;
    using __Tensor::val;
    using __Tensor::shape;
    using __Tensor::sizes;
    using Shape = typename __Tensor::Shape;
    constexpr static std::size_t step = simd::template Step<T>::value;
public:
    Tensorsi_(){}
    /* contruct with shape */
    explicit Tensorsi_(const std::vector<int> &shape_):__Tensor(shape_){}
    explicit Tensorsi_(const std::vector<int> &shape_, const __Tensor &val_)
        :__Tensor(shape_, val_){}
    explicit Tensorsi_(const std::initializer_list<int> &shape_, const std::initializer_list<T> &val_)
        :__Tensor(shape_, val_){}
    /* construct with shape */
    template<typename ...Dim>
    explicit Tensorsi_(Dim ...dim):__Tensor(dim...){}
    /* copy construct */
    Tensorsi_(const __Tensor &r):__Tensor (r){}
    Tensorsi_(const Tensorsi_ &r):__Tensor(r.shape, r.val){}
    /* assign operator */
    Tensorsi_ &operator=(const Tensorsi_ &r)
    {
        if (this == &r) {
            return *this;
        }
        __Tensor::operator=(r);
        return *this;
    }
    Tensorsi_ &operator=(const __Tensor &r)
    {
        if (this == &r) {
            return *this;
        }
        __Tensor::operator=(r);
        return *this;
    }

    Tensorsi_ &operator=(T value)
    {
        __Tensor::operator=(value);
        return *this;
    }

    void zero()
    {
        if (totalSize < step) {
            return __Tensor::zero();
        }
        Instruct::fill(__Tensor::ptr(), 0, totalSize);
        return;
    }
    void fill(T value)
    {
        if (totalSize < step) {
            return __Tensor::fill(value);
        }
        Instruct::fill(__Tensor::ptr(), value, totalSize);
        return;
    }

    static Tensorsi_ zeros(Shape &shape)
    {
        Tensorsi_ x(shape);
        return x;
    }

    template<typename ...Dim>
    static Tensorsi_ zeros(Dim ...dim)
    {
        Tensorsi_ x(dim...);
        return x;
    }

    static Tensorsi_ ones(Shape &shape)
    {
        Tensorsi_ x(shape);
        if (totalSize < step) {
            return __Tensor::fill(1);
        }
        Instruct::fill(__Tensor::ptr(), 1, totalSize);
        return x;
    }

    template<typename ...Index>
    static Tensorsi_ ones(Index ...index)
    {
        Tensorsi_ x(index...);
        Instruct::fill(x.ptr(), 1, x.totalSize);
        return x;
    }

    /* operator */
    Tensorsi_ operator +(const Tensorsi_ &x) const
    {
        Tensorsi_ y(shape);
        if (x.totalSize < step) {
            return __Tensor::operator+(x);
        }
        Instruct::add(y.ptr(), __Tensor::ptr(), x.ptr(), totalSize);
        return y;
    }

    Tensorsi_ operator -(const Tensorsi_ &x) const
    {
        Tensorsi_ y(shape);
        if (x.totalSize < step) {
            return __Tensor::operator-(x);
        }
        Instruct::sub(y.ptr(), __Tensor::ptr(), x.ptr(), totalSize);
        return y;
    }

    Tensorsi_ operator *(const Tensorsi_ &x) const
    {
        Tensorsi_ y(shape);
        if (x.totalSize < step) {
            return __Tensor::operator*(x);
        }
        Instruct::mul(y.ptr(), __Tensor::ptr(), x.ptr(), totalSize);
        return y;
    }

    Tensorsi_ operator /(const Tensorsi_ &x) const
    {
        Tensorsi_ y(shape);
        if (totalSize < step) {
            return __Tensor::operator/(x);
        }
        Instruct::div(y.ptr(), __Tensor::ptr(), x.ptr(), totalSize);
        return y;
    }

    Tensorsi_ operator +(T x) const
    {
        Tensorsi_ y(shape);
        if (totalSize < step) {
            return __Tensor::operator+(x);
        }
        Instruct::add(y.ptr(), __Tensor::ptr(), x, totalSize);
        return y;
    }

    Tensorsi_ operator -(T x) const
    {
        Tensorsi_ y(shape);
        if (totalSize < step) {
            return __Tensor::operator-(x);
        }
        Instruct::sub(y.ptr(), __Tensor::ptr(), x, totalSize);
        return y;
    }

    Tensorsi_ operator *(T x) const
    {
        Tensorsi_ y(shape);
        if (totalSize < step) {
            return __Tensor::operator*(x);
        }
        Instruct::mul(y.ptr(), __Tensor::ptr(), x, totalSize);
        return y;
    }

    Tensorsi_ operator /(T x) const
    {
        Tensorsi_ y(shape);
        if (totalSize < step) {
            return __Tensor::operator/(x);
        }
        Instruct::div(y.ptr(), __Tensor::ptr(), x, totalSize);
        return y;
    }

    Tensorsi_ &operator +=(const Tensorsi_& x)
    {
        if (totalSize < step) {
            __Tensor::operator+=(x);
            return *this;
        }
        Instruct::add(__Tensor::ptr(), __Tensor::ptr(), x.ptr(), totalSize);
        return *this;
    }

    Tensorsi_ &operator -=(const Tensorsi_& x)
    {
        if (totalSize < step) {
            __Tensor::operator-=(x);
            return *this;
        }
        Instruct::sub(__Tensor::ptr(), __Tensor::ptr(), x.ptr(), totalSize);
        return *this;
    }

    Tensorsi_ &operator *=(const Tensorsi_& x)
    {
        if (totalSize < step) {
            __Tensor::operator*=(x);
            return *this;
        }
        Instruct::mul(__Tensor::ptr(), __Tensor::ptr(), x.ptr(), totalSize);
        return *this;
    }

    Tensorsi_ &operator /=(const Tensorsi_& x)
    {
        if (totalSize < step) {
            __Tensor::operator/=(x);
            return *this;
        }
        Instruct::div(__Tensor::ptr(), __Tensor::ptr(), x.ptr(), totalSize);
        return *this;
    }

    Tensorsi_ &operator +=(T x)
    {
        if (totalSize < step) {
            __Tensor::operator+=(x);
            return *this;
        }
        Instruct::add(__Tensor::ptr(), __Tensor::ptr(), x, totalSize);
        return *this;
    }

    Tensorsi_ &operator -=(T x)
    {
        if (totalSize < step) {
            __Tensor::operator-=(x);
            return *this;
        }
        Instruct::sub(__Tensor::ptr(), __Tensor::ptr(), x, totalSize);
        return *this;
    }

    Tensorsi_ &operator *=(T x)
    {
        if (totalSize < step) {
            __Tensor::operator*=(x);
            return *this;
        }
        Instruct::mul(__Tensor::ptr(), __Tensor::ptr(), x, totalSize);
        return *this;
    }

    Tensorsi_ &operator /=(T x)
    {
        if (totalSize < step) {
            __Tensor::operator/=(x);
            return *this;
        }
        Instruct::div(__Tensor::ptr(), __Tensor::ptr(), x, totalSize);
        return *this;
    }
    /* statistics */
    template<typename ...Index>
    T sum(Index ...index) const
    {
        return __Tensor::sum(index...);
    }

    T sum() const
    {
        if (totalSize < step) {
            return __Tensor::sum();
        }
        return Instruct::sum(__Tensor::ptr(), totalSize);
    }

    template<typename ...Index>
    T mean(Index ...index) const
    {
        return __Tensor::mean(index...);
    }

    T mean() const{ return sum()/T(totalSize);}

    template<typename ...Index>
    T variance(T u, Index ...index) const
    {
        return __Tensor::variance(u, index...);
    }

    T variance(T u) const
    {
        if (totalSize < step) {
            return __Tensor::variance(u);
        }
        return Instruct::variance(__Tensor::ptr(), u, totalSize);
    }

    template<typename ...Index>
    T max(Index ...index) const
    {
        return __Tensor::max(index...);
    }

    T max() const
    {
        if (totalSize < step) {
            return __Tensor::max();
        }
        return Instruct::max(__Tensor::ptr(), totalSize);
    }

    template<typename ...Index>
    T min(Index ...index) const
    {
        return __Tensor::min(index...);
    }

    T min() const
    {
        if (totalSize < step) {
            return __Tensor::min();
        }
        return Instruct::min(__Tensor::ptr(), totalSize);
    }

    /* matrix operation */
    struct Mul {
        static void ikkj(Tensorsi_ &x, const Tensorsi_ &x1, const Tensorsi_ &x2)
        {
            /* x = x1 * x2 */
            if (x1.shape[0] < step || x1.shape[1] < step || x2.shape[1] < step) {
                return __Tensor::Mul::ikkj(x, x1, x2);
            }
            Instruct::MatMul::ikkj(x.ptr(), x.shape[0], x.shape[1],
                                   x1.ptr(), x1.shape[0], x1.shape[1],
                                   x2.ptr(), x2.shape[0], x2.shape[1]);
            return;
        }
        static void kikj(Tensorsi_ &x, const Tensorsi_ &x1, const Tensorsi_ &x2)
        {
            /* x = x1^T * x2 */
            if (x1.shape[0] < step || x1.shape[1] < step || x2.shape[1] < step) {
                return __Tensor::Mul::kikj(x, x1, x2);
            }
            /* transpose x1 */
            Instruct::MatMul::kikj(x.ptr(), x.shape[0], x.shape[1],
                                   x1.ptr(), x1.shape[0], x1.shape[1],
                                   x2.ptr(), x2.shape[0], x2.shape[1]);
            return;
        }
        static void ikjk(Tensorsi_ &x, const Tensorsi_ &x1, const Tensorsi_ &x2)
        {
            /* x = x1 * x2^T */
            if (x1.shape[0] < step || x1.shape[1] < step || x2.shape[0] < step) {
                return __Tensor::Mul::ikjk(x, x1, x2);
            }
            /* transpose x2 */
            Instruct::MatMul::ikjk(x.ptr(), x.shape[0], x.shape[1],
                                   x1.ptr(), x1.shape[0], x1.shape[1],
                                   x2.ptr(), x2.shape[0], x2.shape[1]);
            return;
        }
        static void kijk(Tensorsi_ &x, const Tensorsi_ &x1, const Tensorsi_ &x2)
        {
            /* x = x1^T * x2^T */
            if (x1.shape[0] < step || x1.shape[1] < step || x2.shape[0] < step) {
                return __Tensor::Mul::kijk(x, x1, x2);
            }
            /* transpose x1, x2 */
            Instruct::MatMul::kijk(x.ptr(), x.shape[0], x.shape[1],
                                   x1.ptr(), x1.shape[0], x1.shape[1],
                                   x2.ptr(), x2.shape[0], x2.shape[1]);
            return;
        }

    };

};
#if defined(__AVX2__)
using Tensorsi = Tensorsi_<float, simd::AVX2, AlignAllocator32>;
//using Tensorsi = Tensorsi_<float, simd::AVX2, std::allocator>;
#elif defined (__SSE2__)
using Tensorsi = Tensorsi_<float, simd::SSE2, AlignAllocator32>;
#endif
#endif // TENSORSI_HPP
