#ifndef POINT_HPP
#define POINT_HPP
#include <initializer_list>

template <typename T>
class Point2
{
public:
    T x;
    T y;
public:
    Point2():x(0),y(0){}
    Point2(T x_, T y_):x(x_),y(y_){}
    Point2(std::initializer_list<T> list)
    {
        auto it = list.begin();
        x = *it;
        y = *(++it);
    }
    Point2 operator + (const Point2 &r) const {return Point2(x + r.x, y + r.y);}
    Point2 operator - (const Point2 &r) const {return Point2(x - r.x, y - r.y);}
    Point2 operator * (const Point2 &r) const {return Point2(x * r.x, y * r.y);}
    Point2 operator / (const Point2 &r) const {return Point2(x / r.x, y / r.y);}

    Point2& operator += (const Point2 &r) {x += r.x; y += r.y; return *this;}
    Point2& operator -= (const Point2 &r) {x -= r.x; y -= r.y; return *this;}
    Point2& operator *= (const Point2 &r) {x *= r.x; y *= r.y; return *this;}
    Point2& operator /= (const Point2 &r) {x /= r.x; y /= r.y; return *this;}

    Point2 operator + (T c) const {return Point2(x + c, y + c);}
    Point2 operator - (T c) const {return Point2(x - c, y - c);}
    Point2 operator * (T c) const {return Point2(x * c, y * c);}
    Point2 operator / (T c) const {return Point2(x / c, y / c);}

    Point2& operator += (T c) {x += c; y += c; return *this;}
    Point2& operator -= (T c) {x -= c; y -= c; return *this;}
    Point2& operator *= (T c) {x *= c; y *= c; return *this;}
    Point2& operator /= (T c) {x /= c; y /= c; return *this;}
};

template <typename T>
class Point3
{
public:
    T x;
    T y;
    T z;
public:
    Point3():x(0),y(0),z(0){}
    Point3(T x_, T y_, T z_):x(x_),y(y_),z(z_){}
    Point3 operator + (const Point3 &r) const {return Point3(x + r.x, y + r.y, z + r.z);}
    Point3 operator - (const Point3 &r) const {return Point3(x - r.x, y - r.y, z - r.z);}
    Point3 operator * (const Point3 &r) const {return Point3(x * r.x, y * r.y, z * r.z);}
    Point3 operator / (const Point3 &r) const {return Point3(x / r.x, y / r.y, z / r.z);}

    Point3& operator += (const Point3 &r) {x += r.x; y += r.y; z += r.z; return *this;}
    Point3& operator -= (const Point3 &r) {x -= r.x; y -= r.y; z -= r.z; return *this;}
    Point3& operator *= (const Point3 &r) {x *= r.x; y *= r.y; z *= r.z; return *this;}
    Point3& operator /= (const Point3 &r) {x /= r.x; y /= r.y; z /= r.z; return *this;}

    Point3 operator + (T c) const {return Point3(x + c, y + c, z + c);}
    Point3 operator - (T c) const {return Point3(x - c, y - c, z - c);}
    Point3 operator * (T c) const {return Point3(x * c, y * c, z * c);}
    Point3 operator / (T c) const {return Point3(x / c, y / c, z / c);}

    Point3& operator += (T c) {x += c; y += c; z += c; return *this;}
    Point3& operator -= (T c) {x -= c; y -= c; z -= c; return *this;}
    Point3& operator *= (T c) {x *= c; y *= c; z *= c; return *this;}
    Point3& operator /= (T c) {x /= c; y /= c; z /= c; return *this;}
};
using Point2uc = Point2<unsigned char>;
using Point2c = Point2<char>;
using Point2i = Point2<int>;
using Point2f = Point2<float>;
using Point2d = Point2<double>;

using Point3uc = Point3<unsigned char>;
using Point3i = Point3<int>;
using Point3f = Point3<float>;
using Point3d = Point3<double>;

#endif // POINT_HPP
