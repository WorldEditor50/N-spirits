#ifndef COMPLEXNUMBER_H
#define COMPLEXNUMBER_H
#include "basic_def.h"
#include <iostream>
#include <cmath>

class Complex
{
public:
    float re;
    float im;
public:
    Complex():re(0),im(0){}
    explicit Complex(float r, float i)
        :re(r),im(i){}
    explicit Complex(float value):re(value),im(0){}
    Complex(const Complex &r):re(r.re),im(r.im){}
    Complex& operator=(const Complex &r)
    {
        if (this == &r) {
            return *this;
        }
        re = r.re;
        im = r.im;
        return *this;
    }

    Complex(Complex &&r) noexcept
        :re(r.re),im(r.im)
    {
        r.re = 0;
        r.im = 0;
    }
    Complex& operator=(Complex &&r) noexcept
    {
        if (this == &r) {
            return *this;
        }
        re = r.re;
        im = r.im;
        r.re = 0;
        r.im = 0;
        return *this;
    }

    Complex& operator=(float value)
    {
        re = value;
        return *this;
    }
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
    void print() const
    {
        if (im > 0) {
            std::cout<<re<<" + "<<im<<"i"<<std::endl;
        } else if (im == 0) {
            std::cout<<re<<std::endl;
        } else {
            std::cout<<re<<" - "<<std::abs(im)<<"i"<<std::endl;
        }
        return;
    }
};

inline Complex exp(const Complex &c)
{
    /*
        exp(a + ib) = exp(a)*exp(ib) = exp(a)*(cos(b) + isin(b))
    */
    float rho = std::exp(c.re);
    return Complex(rho*std::cos(c.im), rho*std::sin(c.im));
}

inline Complex log(const Complex &c)
{
    /*
        log(a + ib) = log(r*e^(i*t)) = log(r) + i*t
        r = sqrt(a*a + b*b)
        t = argtan(b/a)
    */
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

inline Complex sin(const Complex &c)
{
    /*
        z = a + ib
        e^(i*z)  = cos(z) + i*sin(z)
        e^(-i*z) = cos(z) - i*sin(z)
        sin(z) = (e^(i*z) - e^(-i*z))/2i
        e^(i*z)  = e^(i*(a + ib)) = e^(-b + ia)
        e^(-i*z) = e^(-i*(a + ib) = e^(b - ia)
        sin(a + ib) = (e^(-b + ia) - e^(b - ia))/2i
                    = (e^(-b)*(cos(a) + i*sin(a)) - e^b*(cos(a) -i*sin(a)))/2i
                    = (e^(-b)*cos(a) - e^b*cos(a))/2i + (e^(-b)*sin(a) + e^b*sin(a))/2i
                    = (-sinh(b)cos(a) + i*cosh(b)sin(a))/i
                    = cosh(b)sin(a) + i*sinh(b)cos(a)
    */
    return Complex(std::sin(c.re)*std::cosh(c.im), std::sinh(c.im)*std::cos(c.re));
}

inline Complex cos(const Complex &c)
{
    /*
        z = a + ib
        e^(i*z)  = cos(z) + i*sin(z)
        e^(-i*z) = cos(z) - i*sin(z)
        cos(z) = (e^(i*z) + e^(-i*z))/2
        e^(i*z)  = e^(i*(a + ib)) = e^(-b + ia)
        e^(-i*z) = e^(-i*(a + ib) = e^(b - ia)
        cos(a + ib) = (e^(-b + ia) + e^(b - ia))/2
                    = (e^(-b)*e^(ia) + e^b*e^(-ia))/2
                    = (e^(-b)*(cos(a) + i*sin(a)) + e^b*(cos(a) -i*sin(a)))/2
                    = (e^(-b) + e^b)/2 * cos(a) + i*(e^(-b) - e^b)/2*sin(a)
                    = (e^b + e^(-b))/2 * cos(a) - i*(e^b - e^(-b))/2*sin(a)
                    = cosh(b)*cos(a) - i*sinh(b)*sin(a)
    */
    return Complex(std::cos(c.re)*std::cosh(c.im), -1*std::sin(c.im)*std::sinh(c.re));
}

inline Complex tan(const Complex &c)
{
    /*
        tan(a + ib) = sin(a + ib)/cos(a + ib)
                    = (cosh(b)sin(a) + i*sinh(b)cos(a))/(cosh(b)*cos(a) - i*sinh(b)*sin(a))
        g = cosh(b)*cos(a)
        h = sinh(b)*sin(a)
        p = sin(a)*cosh(b)
        q = sinh(b)*cos(a)

        tan(a + ib) = (p + iq)/(g - ih)
                    = (p + iq)*(g + ih)/(g^2 + h^2)
                    = (pg - qh + i(qg + ph))/(g^2 + h^2)
                    = (pg - qh)/(g^2 + h^2) + i * (qg + ph)/(g^2 + h^2)


    */
    float a = c.re;
    float b = c.im;
    float g = std::cosh(b)*std::cos(a);
    float h = std::sinh(b)*std::sin(a);
    float p = std::sin(a)*std::cosh(b);
    float q = std::sinh(b)*std::cos(a);
    float r = g*g + h*h;
    return Complex((p*g - q*h)/r, (q*g + p*h)/r);
}

#endif // COMPLEXNUMBER_H
