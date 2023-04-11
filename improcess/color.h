#ifndef COLOR_H
#define COLOR_H
#include <cstdint>

namespace imp {

class Color3
{
public:
    uint8_t r;
    uint8_t g;
    uint8_t b;
public:
    Color3():r(0),g(0),b(0){}
    explicit Color3(uint8_t r_, uint8_t g_, uint8_t b_)
        :r(r_),g(g_),b(b_){}
};

}

#endif // COLOR_H
