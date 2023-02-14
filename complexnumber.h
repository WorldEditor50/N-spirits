#ifndef COMPLEXNUMBER_H
#define COMPLEXNUMBER_H
#include <cmath>
#include "basic_def.h"

class Complex
{
public:
    float re;
    float im;
public:
    Complex():re(0),im(0){}
    explicit Complex(float r, float i)
        :re(r),im(i){}
    static Complex fromPolor(float rho, float theta)
    {
        Complex c;
        c.re = rho*std::cos(theta);
        c.im = rho*std::sin(theta);
        return c;
    }
    inline float modulus() const
    {
        return std::sqrt(re*re + im*im);
    }
    inline float arg() const
    {
        return std::atan(im/re);
    }
    inline Complex conjugate() const
    {
        return Complex(re, -im);
    }
    inline Complex operator + (const Complex &c) const
    {
        return Complex(re + c.re, im + c.im);
    }
    inline Complex operator - (const Complex &c) const
    {
        return Complex(re - c.re, im - c.im);
    }
    inline Complex operator * (const Complex &c) const
    {
        return Complex(re*c.re - im*c.im, re*c.im + im*c.re);
    }
    inline Complex operator / (const Complex &c) const
    {
        Complex y;
        float rho = c.modulus();
        y.re = (re*c.re + im*c.im)/rho;
        y.im = (re*c.im - im*c.re)/rho;
        return y;
    }

    inline Complex operator + (float r) const
    {
        return Complex(re + r, im);
    }
    inline Complex operator - (float r) const
    {
        return Complex(re - r, im);
    }
    inline Complex operator * (float r) const
    {
        return Complex(re*r, im*r);
    }
    inline Complex operator / (float r) const
    {
        return Complex(re/r, im/r);
    }

    inline Complex& operator += (const Complex &c)
    {
        re += c.re;
        im += c.im;
        return *this;
    }
    inline Complex& operator -= (const Complex &c)
    {
        re -= c.re;
        im -= c.im;
        return *this;
    }
    inline Complex& operator *= (const Complex &c)
    {
        Complex c_(re, im);
        re = c_.re*c.re - c_.im*c.im;
        im = c_.re*c.im + c_.im*c.re;
        return *this;
    }
    inline Complex& operator /= (const Complex &c)
    {
        Complex y;
        float rho = c.modulus();
        y.re = (re*c.re + im*c.im)/rho;
        y.im = (re*c.im - im*c.re)/rho;
        re = y.re;
        im = y.im;
        return *this;
    }

    inline Complex& operator += (float r)
    {
        re += r;
        return *this;
    }
    inline Complex& operator -= (float r)
    {
        re -= r;
        return *this;
    }
    inline Complex& operator *= (float r)
    {
        re *= r;
        im *= r;
        return *this;
    }
    inline Complex& operator /= (float r)
    {
        re /= r;
        im /= r;
        return *this;
    }
};

inline Complex exp(const Complex &c)
{
    float rho = std::exp(c.re);
    return Complex(rho*std::cos(c.im), rho*std::sin(c.im));
}

inline Complex log(const Complex &c)
{
    float rho = c.modulus();
    float theta = c.arg();
    return Complex(rho, theta);
}

inline Complex sqrt(const Complex &c)
{
    float rho = c.modulus();
    float theta = c.arg()/2;
    return Complex(rho*std::cos(theta), rho*std::sin(theta));
}

inline Complex pow(const Complex &c, float n)
{
    float rho = c.modulus();
    float theta = c.arg()*n;
    return Complex(rho*std::cos(theta), rho*std::sin(theta));
}

inline Complex pow(float r, const Complex & n)
{
    float rho = std::exp(n.re*std::log(r));
    return Complex(rho*std::cos(n.im), rho*std::sin(n.im));
}

inline Complex pow(const Complex r, const Complex & n)
{
    float rho = r.modulus()*std::exp(-n.im);
    float theta = r.arg()*n.re;
    return Complex(rho*std::cos(theta), rho*std::sin(theta));
}


inline Complex sinh(const Complex &c)
{
    return (exp(c) - exp(c*(-1)))/2;
}

inline Complex cosh(const Complex &c)
{
    return (exp(c) + exp(c*(-1)))/2;
}

inline Complex tanh(const Complex &c)
{
    Complex c1 = exp(c);
    Complex c2 = exp(c*(-1));
    return (c1 - c2)/(c1 + c2);
}
#endif // COMPLEXNUMBER_H
