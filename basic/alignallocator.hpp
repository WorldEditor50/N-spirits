#ifndef ALIGNALLOCATOR_HPP
#define ALIGNALLOCATOR_HPP
#include <memory>

template <typename T, std::size_t N>
class AlignedAllocator {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
public:
    inline AlignedAllocator() throw() {}
    template <typename T2>
    inline AlignedAllocator(const AlignedAllocator<T2, N> &) throw() {}
    inline ~AlignedAllocator() throw() {}
    inline pointer adress(reference r) { return &r; }
    inline const_pointer adress(const_reference r) const { return &r; }
    inline pointer allocate(size_type n);
    inline void deallocate(pointer p, size_type);
    inline void construct(pointer p, const value_type & v) { new (p) value_type(v); }
    inline void destroy(pointer p) { p->~value_type(); }
    inline size_type max_size() const throw() { return size_type(-1) / sizeof(value_type); }
    template <typename T2>
    struct rebind {
        using other = AlignedAllocator<T2, N>;
    };
    bool operator!=(const AlignedAllocator<T, N> & other) const { return !(*this == other); }
    bool operator==(const AlignedAllocator<T, N> & other) const { return true; }
};

template <typename T, std::size_t N>
inline typename AlignedAllocator<T, N>::pointer AlignedAllocator<T, N>::allocate(size_type n)
{
#ifdef _MSC_VER
    auto p = (pointer)_aligned_malloc(n * sizeof(value_type), N);
#else
    auto p = (pointer)aligned_alloc(N, n * sizeof(value_type));
#endif
    if (p == nullptr) {
        throw std::bad_alloc();
    }
    return p;
}

template <typename T, std::size_t N>
inline void AlignedAllocator<T, N>::deallocate(pointer p, size_type)
{
#ifdef _MSC_VER
    _aligned_free(p);
#else
    std::free(p);
#endif
    return;
}

template <typename T>
using AlignAllocator16 = AlignedAllocator<T, 16>;

template <typename T>
using AlignAllocator32 = AlignedAllocator<T, 32>;

template <typename T>
using AlignAllocator64 = AlignedAllocator<T, 64>;

#endif // ALIGNALLOCATOR_HPP
