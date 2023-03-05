#ifndef IMPROCESS_H
#define IMPROCESS_H
#include <string>
#include <memory>
#include <functional>
#include "jpegwrap.h"
#include "image.hpp"


namespace improcess {

    /* tensor shape: (h, w, c) */
    int load(const std::string &fileName, Tensor &img);
    int save(const std::string &fileName, const Tensor &img);
    /* gray */
    int convert2Gray(const Tensor& rgb, Tensor &gray);

}


#endif // IMPROCESS_H

