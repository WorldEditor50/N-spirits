#ifndef SIMD_HPP
#define SIMD_HPP
#include <immintrin.h>
#include <vector>
#include <cmath>
#include "sse2func.hpp"
#include "avx2func.hpp"

namespace simd {

struct None {
    struct Unit {
        constexpr static std::size_t value = 0;
    };
};

template<std::size_t Ni, typename Instruction=SSE2,typename T=double, std::size_t align=16>
class Vector
{
public:
    using Type = T;
    constexpr static std::size_t N = Ni;
public:
    T* val;
public:
    Vector():val(nullptr)
    {
        val = static_cast<T*>(_mm_malloc(sizeof (double)*N, align));
    }
    ~Vector()
    {
        if (val != nullptr) {
            _mm_free(val);
            val = nullptr;
        }
    }

    inline std::size_t size() const {return N;}

    Vector operator + (const Vector &x) const
    {
        Vector<N> y;
        Instruction::add(y.val, val, x.val, N);
        return y;
    }

    Vector operator - (const Vector &x) const
    {
        Vector<N> y;
        Instruction::sub(y.val, val, x.val, N);
        return y;
    }

    Vector operator * (const Vector &x) const
    {
        Vector<N> y;
        Instruction::mul(y.val, val, x.val, N);
        return y;
    }

    Vector operator / (const Vector &x) const
    {
        Vector<N> y;
        Instruction::div(y.val, val, x.val, N);
        return y;
    }
};


}


#endif // SIMD_HPP
