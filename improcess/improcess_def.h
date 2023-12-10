#ifndef IMPROCESS_DEF_H
#define IMPROCESS_DEF_H
#include "../basic/tensor.hpp"
#include <memory>

using u8ptr = std::unique_ptr<uint8_t>;

namespace imp {

inline double bound(double x, double min_, double max_)
{
    double value = x < min_ ? min_ : x;
    value = value > max_ ? max_ : x;
    return value;
}

}

#endif // IMPROCESS_DEF_H
