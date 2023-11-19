#ifndef LIME_HPP
#define LIME_HPP
#include <vector>
#include "../basic/tensor.hpp"
#include "../basic/util.hpp"

class Lime
{
public:

public:
    Lime()
    {

    }

    Tensor explain(const std::vector<Tensor> &data, std::size_t numberOfSample, float sigma)
    {
        Tensor weight;
        /* step1: sample around */

        /* step2: generate label */
        /* step3: calculate distance */
        /* step4: explain */

        return weight;
    }

};

#endif // LIME_HPP
