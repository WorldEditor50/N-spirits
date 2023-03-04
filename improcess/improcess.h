#ifndef IMPROCESS_H
#define IMPROCESS_H
#include <string>
#include <memory>
#include "../basic/tensor.hpp"
#include "jpegwrap.h"

namespace improcess {

enum HWC {
    HWC_H = 0,
    HWC_W = 1,
    HWC_C = 2
};

    /* tensor shape: (h, w, c) */
    int load(const std::string &fileName, Tensor &img);
    int save(const std::string &fileName, const Tensor &img);
    /* gray */
    int convert2Gray(const Tensor& rgb, Tensor &gray);
}


#endif // IMPROCESS_H

