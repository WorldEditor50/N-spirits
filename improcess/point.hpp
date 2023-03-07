#ifndef Point2_HPP
#define Point2_HPP

template <typename T>
class Point2
{
public:
    T x;
    T y;
public:
    Point2():x(0),y(0){}
    Point2(T x_, T y_):x(x_),y(y_){}
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

using Point2i = Point2<int>;
using Point2f = Point2<float>;
using Point2d = Point2<double>;

#endif // Point2_HPP
