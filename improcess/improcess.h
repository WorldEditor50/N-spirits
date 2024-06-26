#ifndef IMPROCESS_H
#define IMPROCESS_H
#include <string>
#include <memory>
#include <functional>
#include "improcess_def.h"
#include "bmp.hpp"
#include "ppm.hpp"
#include "image.hpp"
#include "graphic2d.h"
#include "geometrytransform.h"
#include "filter.h"
#include "features.h"
#ifdef ENABLE_JPEG
#include "jpegwrap/jpegwrap.h"
#endif

#ifdef WIN32
#include "platform/windows/viewpage.h"
#endif

#ifdef __linux__
#include "platform/linux/viewpage.h"
#endif

namespace imp {

    /* tensor shape: (h, w, c) */
    Tensor load(const std::string &fileName);
    int save(InTensor img, const std::string &fileName);
    Tensor toTensor(int h, int w, int c, std::shared_ptr<uint8_t[]> &img);
    int fromTensor(InTensor x, std::shared_ptr<uint8_t[]> &img);
    std::unique_ptr<uint8_t[]> fromTensor(InTensor x);
    std::shared_ptr<uint8_t[]> tensor2Image(InTensor x);
    /* show image */
    void showHistogram(InTensor x);
    void show(InTensor x);
    void show(const std::string &fileName);
    /* copy */
    int copyMakeBorder(OutTensor xo, InTensor xi, int padding);
    int copy(OutTensor &xo, InTensor xi, const Rect &rect);
    /* gray */
    int rgb2gray(OutTensor gray, InTensor rgb);
    int gray2rgb(OutTensor rgb, InTensor gray);
    /* rgb <--> rgba */
    int rgb2rgba(OutTensor rgba, InTensor rgb, int alpha=120);
    int rgba2rgb(OutTensor rgba, InTensor rgb);
    int transparent(OutTensor rgba, InTensor rgb, int alpha=120);
    /* resize */
    int resize(OutTensor xo, InTensor xi, const Size &size, int type=INTERPOLATE_NEAREST);
    /* erode */
    int erode(OutTensor xo, InTensor xi, InTensor kernel);
    /* dilate */
    int dilate(OutTensor xo, InTensor xi, InTensor kernel);
    /* gray dilate */
    int grayDilate(OutTensor xo, const Point2i &offset, InTensor xi, InTensor kernel);
    /* gray erode */
    int grayErode(OutTensor xo, const Point2i &offset, InTensor xi, InTensor kernel);
    /* trace boundary */
    int traceBoundary(OutTensor xo, InTensor xi, std::vector<Point2i> &boundary);
    /* connected region */
    int findConnectedRegion(OutTensor mask, InTensor xi, int connectCount, int &labelCount);
    /* threshold */
    int threshold(OutTensor xo, InTensor xi, float thres, float max_, float min_);
    /* detect threshold */
    int detectThreshold(InTensor xi, int maxIter, int &thre, int &delta);
    /* auto threshold */
    int autoThreshold(OutTensor xo, InTensor xi, float max_, float min_);
    /* otsu threshold */
    int otsuThreshold(OutTensor xo, InTensor xi, float max_, float min_);
    /* entropy threshold */
    int entropyThreshold(OutTensor xo, InTensor xi, float max_, float min_);
    /* region grow */
    int regionGrow(OutTensor mask, float label, InTensor xi,  const Point2i &seed, uint8_t thres);
    /* template match */
    int templateMatch(InTensor xi, InTensor xt, Rect &rect);
}


#endif // IMPROCESS_H

