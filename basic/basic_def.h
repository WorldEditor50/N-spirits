#ifndef BASIC_DEF_H
#define BASIC_DEF_H
constexpr static float pi = 3.14159;

#ifdef _MSC_VER
    #define FORCE_INLINE __forceinline
    #define VECTORCALL   __vectorcall
#else
    #define FORCE_INLINE __attribute__((always_inline)) inline
    #define VECTORCALL
#endif

#endif // BASIC_DEF_H
