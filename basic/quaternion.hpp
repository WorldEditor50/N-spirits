#ifndef QUATERNION_HPP
#define QUATERNION_HPP
#include <cmath>

struct Quad {
    float x0;
    float xi;
    float xj;
    float xk;
};

class Quaternion
{
public:
    float x0;
    float xi;
    float xj;
    float xk;
public:
    Quaternion()
        :x0(0),xi(0),xj(0),xk(0){}
    Quaternion(float x0_, float xi_, float xj_, float xk_)
        :x0(x0_),xi(xi_),xj(xj_),xk(xk_){}
    Quaternion(float xi_, float xj_, float xk_)
        :x0(0),xi(xi_),xj(xj_),xk(xk_){}
    Quaternion(const Quaternion& r)
        :x0(r.x0),xi(r.xi),xj(r.xj),xk(r.xk){}
    Quaternion& operator=(const Quaternion& r)
    {
        if (this == &r) {
            return *this;
        }
        x0 = r.x0;
        xi = r.xi;
        xj = r.xj;
        xk = r.xk;
        return *this;
    }

    bool operator==(const Quaternion& r)
    {
        if (this == &r) {
            return true;
        }
        return (x0==r.x0)&&(xi==r.xi)&&(xj==r.xj)&&(xk==r.xk);
    }

    Quaternion& operator+=(const Quaternion& r)
    {
        x0 += r.x0;
        xi += r.xi;
        xj += r.xj;
        xk += r.xk;
        return *this;
    }
    Quaternion& operator-=(const Quaternion& r)
    {
        x0 -= r.x0;
        xi -= r.xi;
        xj -= r.xj;
        xk -= r.xk;
        return *this;
    }
    Quaternion& operator+=(float r)
    {
        x0 += r;
        return *this;
    }
    Quaternion& operator-=(float r)
    {
        x0 -= r;
        return *this;
    }
    Quaternion& operator*=(float r)
    {
        x0 *= r;
        xi *= r;
        xj *= r;
        xk *= r;
        return *this;
    }
    Quaternion& operator/=(float r)
    {
        x0 /= r;
        xi /= r;
        xj /= r;
        xk /= r;
        return *this;
    }
    Quaternion operator+(const Quaternion& r)
    {
        Quaternion q;
        q.x0 = x0 + r.x0;
        q.xi = xi + r.xi;
        q.xj = xj + r.xj;
        q.xk = xk + r.xk;
        return q;
    }
    Quaternion operator-(const Quaternion& r)
    {
        Quaternion q;
        q.x0 = x0 - r.x0;
        q.xi = xi - r.xi;
        q.xj = xj - r.xj;
        q.xk = xk - r.xk;
        return q;
    }
    Quaternion operator*(const Quaternion& r)
    {
        /*
            i*i = -1
            j*j = -1
            k*k = -1
            i*j = k
            j*k = i
            k*i = j
            i*k = -j
            k*j = -i
            j*i = -k
            q = (x0 + xi + xj + xk)*(r.x0 + r.xi + r.xj + r.xk)
            q.x0 = x0*r.x0 - xi*r.xi - xj*r.xj - xk*r.xk
            q.xi = x0*r.xi + xi*r.x0 + xj*r.xk - xk*r.xj
            q.xj = x0*r.xj + xj*r.x0 + xk*r.xi - xi*r.xk
            q.xk = x0*r.xk + xk*r.xk + xi*r.xk - xj*r.xi
        */
        Quaternion q;
        q.x0 = x0*r.x0 - xi*r.xi - xj*r.xj - xk*r.xk;
        q.xi = x0*r.xi + xi*r.x0 + xj*r.xk - xk*r.xj;
        q.xj = x0*r.xj + xj*r.x0 + xk*r.xi - xi*r.xk;
        q.xk = x0*r.xk + xk*r.x0 + xi*r.xk - xj*r.xi;
        return q;
    }
    Quaternion operator+(float r)
    {
        Quaternion q;
        q.x0 = x0 + r;
        return q;
    }
    Quaternion operator-(float r)
    {
        Quaternion q;
        q.x0 = x0 - r;
        return q;
    }
    Quaternion operator*(float r)
    {
        Quaternion q;
        q.x0 = x0 * r;
        q.xi = xi * r;
        q.xj = xj * r;
        q.xk = xk * r;
        return q;
    }
    Quaternion operator/(float r)
    {
        Quaternion q;
        q.x0 = x0 / r;
        q.xi = xi / r;
        q.xj = xj / r;
        q.xk = xk / r;
        return q;
    }
    Quaternion& operator-()
    {
        x0 = -x0;
        xi = -xi;
        xj = -xj;
        xk = -xk;
        return *this;
    }

    float norm2() const
    {
        return std::sqrt(x0*x0 + xi*xi + xj*xj + xk*xk);
    }

    Quaternion conjugate() const
    {
        return Quaternion(x0, -xi, -xj, -xk);
    }

    Quaternion inverse() const
    {
        float r = x0*x0 + xi*xi + xj*xj + xk*xk;
        return Quaternion(x0/r, -xi/r, -xj/r, -xk/r);
    }
    static float dot(const Quaternion &q1, const Quaternion &q2)
    {
        return q1.x0*q2.x0 + q1.xi*q2.xi + q1.xj*q2.xj + q1.xk*q2.xk;
    }
    static Quaternion slerp(Quaternion &q1, Quaternion &q2, float t)
    {
        float d = dot(q1, q2);
        if (d < 0) {
            q2 = -q2;
            d = -d;
        }
        float k1;
        float k2;
        if (d > 0.9995f) {
            k1 = 1 - t;
            k2 = t;
        } else {
            float theta = std::acos(d);
            k1 = std::sin(1 - t)*theta/std::sin(theta);
            k2 = std::sin(t*theta)/std::sin(theta);
        }
        return q1*k1 + q2*k2;
    }
};















#endif // QUATERNION_HPP
