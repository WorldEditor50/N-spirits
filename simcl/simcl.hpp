#ifndef SIMCL_HPP
#define SIMCL_HPP
#include "simcl.h"
#include "evaluate.hpp"

namespace simcl {

template<typename Op>
struct Compute {
    template<typename T>
    inline static int eval(T *x, T *x1, T *x2, std::size_t totalsize)
    {
        Op op;
        return op(x, x1, x2, totalsize);
    }
    template<typename T>
    inline static int eval(T *y, T *x, std::size_t totalsize)
    {
        Op op;
        return op(y, x, totalsize);
    }
};

}
#endif // SIMCL_HPP
