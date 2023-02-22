#ifndef METAUTIL_H
#define METAUTIL_H
#include <tuple>
#include <type_traits>

template<typename Tuple, typename Func, std::size_t ...Index>
inline void invokeTuple(const Tuple& tuple, Func&& func, std::index_sequence<Index...>)
{
    static_cast<void>(std::initializer_list<int>{(func(std::get<Index>(tuple)), 0)...});
    return;
}
template <typename ... Args, typename Func>
inline void foreachTuple(const std::tuple<Args...>& tuple, Func&& func)
{
    invokeTuple(tuple, std::forward<Func>(func), std::make_index_sequence<sizeof...(Args)>{});
    return;
}
#endif // METAUTIL_H
