#include <chrono>


class Clock
{
public:
    static std::chrono::system_clock::time_point tiktok()
    {
        return std::chrono::system_clock::now();

    }
    static double duration(std::chrono::system_clock::time_point t1,
                           std::chrono::system_clock::time_point t2)
    {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t2);
        return double(duration.count()) * std::chrono::microseconds::period::num /
                            std::chrono::microseconds::period::den;
    }
};
