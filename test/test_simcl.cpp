#include <iostream>
#include <vector>
#include <algorithm>
#include "../simcl/simcl.hpp"

int main()
{
    int ret = simcl::Device::open(simcl::NVIDIA);
    if (ret != 0) {
        std::cout<<"init error = "<<ret<<std::endl;
        return -1;
    }
    std::vector<float> x1(1024, 2);
    std::vector<float> x2(1024, 3);
    std::vector<float> x(1024, 0);
    ret = simcl::Compute<simcl::Mul>::eval(x.data(), x1.data(), x2.data(), 1024);
    if (ret != 0) {
        std::cout<<"compute error="<<std::endl;
        return -2;
    }
    for (std::size_t i = 0; i < 1024; i++) {
        std::cout<<x[i]<<" ";
    }
    std::cout<<std::endl;
    return 0;
}
