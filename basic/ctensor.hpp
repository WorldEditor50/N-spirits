#ifndef CTENSOR_HPP
#define CTENSOR_HPP
#include "tensor.hpp"
#include "complex.hpp"

class CTensor : public Tensor_<Complex>
{
public:
    CTensor(){}
    CTensor(const Tensor_<Complex> &r)
        :Tensor_<Complex>(r){}
    /* contruct with shape */
    explicit CTensor(const Shape &shape_)
        :Tensor_<Complex>(shape_){}
    explicit CTensor(const Shape &shape_, const std::vector<Complex> &val_)
        :Tensor_<Complex>(shape_, val_){}
    explicit CTensor(const std::initializer_list<int> &shape_,
                     const std::initializer_list<Complex> &val_)
        :Tensor_<Complex>(shape_, val_){}
    template<typename ...Dim>
    explicit CTensor(Dim ...dim)
        :Tensor_<Complex>(dim...){}

    CTensor operator +(const Tensorf &x) const
    {
        CTensor y(shape);
        for (std::size_t i = 0; i < val.size(); i++) {
            y.val[i] = val[i].re + x.val[i];
        }
        return y;
    }

    CTensor operator -(const Tensorf &x) const
    {
        CTensor y(shape);
        for (std::size_t i = 0; i < val.size(); i++) {
            y.val[i] = val[i].re - x.val[i];
        }
        return y;
    }

    CTensor operator *(const Tensorf &x) const
    {
        CTensor y(shape);
        for (std::size_t i = 0; i < val.size(); i++) {
            y.val[i] = val[i].re * x.val[i];
            y.val[i] = val[i].im * x.val[i];
        }
        return y;
    }

    CTensor operator /(const Tensorf &x) const
    {
        CTensor y(shape);
        for (std::size_t i = 0; i < val.size(); i++) {
            y.val[i] = val[i].re / x.val[i];
            y.val[i] = val[i].im / x.val[i];
        }
        return y;
    }

    CTensor &operator +=(const Tensorf &x)
    {
        for (std::size_t i = 0; i < val.size(); i++) {
            val[i].re += x.val[i];
        }
        return *this;
    }

    CTensor &operator -=(const Tensorf &x)
    {
        for (std::size_t i = 0; i < val.size(); i++) {
            val[i].re -= x.val[i];
        }
        return *this;
    }

    CTensor &operator *=(const Tensorf &x)
    {
        for (std::size_t i = 0; i < val.size(); i++) {
            val[i].re *= x.val[i];
            val[i].im *= x.val[i];
        }
        return *this;
    }

    CTensor operator /=(const Tensorf &x)
    {
        for (std::size_t i = 0; i < val.size(); i++) {
            val[i].re /= x.val[i];
            val[i].im /= x.val[i];
        }
        return *this;
    }

    CTensor row(int i) const
    {
        CTensor r(shape[1]);
        int pos = i*shape[1];
        for (int j = 0; j < shape[1]; j++) {
            r[j] = val[j + pos];
        }
        return r;
    }

    void row(int i, const CTensor &r)
    {
        int pos = i*shape[1];
        for (int j = 0; j < shape[1]; j++) {
            val[j + pos] = r[j];
        }
        return;
    }

    CTensor column(int j) const
    {
        CTensor c(shape[0]);
        for (int i = 0; i < shape[0]; i++) {
            c[i] = val[i*shape[1] + j];
        }
        return c;
    }

    void column(int j, const CTensor &c)
    {
        for (int i = 0; i < shape[0]; i++) {
            val[i*shape[1] + j] = c[i];
        }
        return;
    }

    /* conjugate transpose */
    CTensor dagger() const
    {
        CTensor x(shape);
        for (std::size_t i = 0; i< totalSize; i++) {
            x[i] = val[i].conjugate();
        }
        return x.tr();
    }

    CTensor unitary() const
    {
        CTensor x(shape);
        return x;
    }

    void printValue() const
    {
        std::cout<<"[";
        for (std::size_t i = 0; i < val.size(); i++) {
            std::cout<<val[i].re<<" + "<<val[i].im<<"i";
            if (i < totalSize - 1) {
                std::cout<<",";
            }
        }
        std::cout<<"]"<<std::endl;
        return;
    }
};
#endif // CTENSOR_HPP
