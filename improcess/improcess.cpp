#include "improcess.h"
#ifdef ENABLE_JPEG
#include "jpegwrap/jpegwrap.h"
#endif

#ifdef WIN32
#include "platform/windows/viewpage.h"
#endif

#ifdef __linux__
#include "platform/linux/viewpage.h"
#endif

int ns::fromTensor(InTensor x, std::shared_ptr<uint8_t[]> &img)
{
    if (img == nullptr) {
        img = std::shared_ptr<uint8_t[]>(new uint8_t[x.totalSize]);
    }
    for (std::size_t i = 0; i < x.totalSize; i++) {
        img[i] = ns::clip(x.val[i], 0, 255);
    }
    return 0;
}

std::unique_ptr<uint8_t[]> ns::fromTensor(InTensor x)
{
    std::unique_ptr<uint8_t[]> img(new uint8_t[x.totalSize]);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        img[i] = ns::clip(x.val[i], 0, 255);
    }
    return img;
}

std::shared_ptr<uint8_t[]> ns::tensor2Image(InTensor x)
{
    std::shared_ptr<uint8_t[]> img(new uint8_t[x.totalSize]);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        img[i] = ns::clip(x.val[i], 0, 255);
    }
    return img;
}

void ns::showHistogram(InTensor xi)
{
    Tensor x;
    if (xi.shape[HWC_C] != 1) {
        rgb2gray(x, xi);
    } else {
        x = xi;
    }
    Tensor hist;
    histogram(hist, x);
    float r = 512.0/hist.max();
    hist *= r;
    Tensor img(512, 512, 3);
    img.fill(255);
    int h = img.shape[HWC_H];
    int w = img.shape[HWC_W];
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            if (i <= hist[j/2]) {
                img(h - i - 1, j, 0) = 0;
                img(h - i - 1, j, 1) = 191;
                img(h - i - 1, j, 2) = 255;
            }
        }
    }
    ns::show(img);
    return;
}

void ns::show(InTensor xi)
{
    if (xi.empty()) {
        return;
    }
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    int c = xi.shape[HWC_C];
    std::shared_ptr<uint8_t[]> img = tensor2Image(xi);
    View2D view;
    view.display(h, w, c, img.get());
    return;
}

void ns::show(const std::string &fileName)
{
    Tensor img = load(fileName);
    show(img);
    return;
}

int ns::copyMakeBorder(OutTensor xo, InTensor xi, int padding)
{
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    int c = xi.shape[HWC_C];
    xo = Tensor(h + 2*padding, w + 2*padding, c);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            for (int k = 0; k < c; k++) {
                xo(i + padding, j + padding, k) = xi(i, j, k);
            }
        }
    }
    return 0;
}


int ns::copy(OutTensor &xo, InTensor xi, const ns::Rect &rect)
{
    int c = xi.shape[HWC_C];
    xo = Tensor(rect.height, rect.width, c);
    for (int i = 0; i < rect.height; i++) {
        for (int j = 0; j < rect.width; j++) {
            for (int k = 0; k < c; k++) {
                xo(i, j, k) = xi(i + rect.x, j + rect.y, k);
            }
        }
    }
    return 0;
}

int ns::copy(OutTensor &xo, InTensor xi, InTensor mask)
{
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    int c = xi.shape[HWC_C];
    xo = Tensor(h, w, c);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            for (int k = 0; k < c; k++) {
                if (mask(i, j, 0)) {
                    xo(i, j, k) = xi(i, j, k);
                }
            }
        }
    }
    return 0;
}

int ns::rgb2gray(OutTensor gray, InTensor rgb)
{
    if (rgb.shape[HWC_C] != 3) {
        return -1;
    }
    int h = rgb.shape[HWC_H];
    int w = rgb.shape[HWC_W];
    gray = Tensor(h, w, 1);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float avg = (rgb(i, j, 0) + rgb(i, j, 1) + rgb(i, j, 2))/3;
            gray(i, j, 0) = avg;
        }
    }
    return 0;
}

int ns::maxGray(OutTensor gray, InTensor rgb)
{
    if (rgb.shape[HWC_C] != 3) {
        return -1;
    }
    int h = rgb.shape[HWC_H];
    int w = rgb.shape[HWC_W];
    gray = Tensor(h, w, 1);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float p = std::max(rgb(i, j, 0),
                               std::max(rgb(i, j, 1), rgb(i, j, 2)));
            gray(i, j, 0) = p;
        }
    }
    return 0;
}

int ns::minGray(OutTensor gray, InTensor rgb)
{
    if (rgb.shape[HWC_C] != 3) {
        return -1;
    }
    int h = rgb.shape[HWC_H];
    int w = rgb.shape[HWC_W];
    gray = Tensor(h, w, 1);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float p = std::min(rgb(i, j, 0),
                               std::min(rgb(i, j, 1), rgb(i, j, 2)));
            gray(i, j, 0) = p;
        }
    }
    return 0;
}
int ns::meanGray(OutTensor gray, InTensor rgb)
{
    if (rgb.shape[HWC_C] != 3) {
        return -1;
    }
    int h = rgb.shape[HWC_H];
    int w = rgb.shape[HWC_W];
    gray = Tensor(h, w, 1);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float p = rgb(i, j, 0)*0.3 + rgb(i, j, 1)*0.59 + rgb(i, j, 2)*0.11;
            gray(i, j, 0) = p;
        }
    }
    return 0;
}

int ns::gray2rgb(OutTensor rgb, InTensor gray)
{
    if (gray.shape[HWC_C] != 1) {
        return -1;
    }
    int h = gray.shape[HWC_H];
    int w = gray.shape[HWC_W];
    rgb = Tensor(h, w, 3);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float pixel = gray(i, j, 0);
            rgb(i, j, 0) = pixel;
            rgb(i, j, 1) = pixel;
            rgb(i, j, 2) = pixel;
        }
    }
    return 0;
}


int ns::rgb2rgba(OutTensor rgba, InTensor rgb, int alpha)
{
    if (rgb.shape[HWC_C] != 3) {
        return -1;
    }
    int h = rgb.shape[HWC_H];
    int w = rgb.shape[HWC_W];
    rgba = Tensor(h, w, 4);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            rgba(i, j, 0) = rgb(i, j, 0);
            rgba(i, j, 1) = rgb(i, j, 1);
            rgba(i, j, 2) = rgb(i, j, 2);
            rgba(i, j, 3) = alpha%256;
        }
    }
    return 0;
}

int ns::rgba2rgb(OutTensor rgb, InTensor rgba)
{
    int h = rgb.shape[HWC_H];
    int w = rgb.shape[HWC_W];
    rgb = Tensor(h, w, 3);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            rgb(i, j, 0) = rgba(i, j, 0);
            rgb(i, j, 1) = rgba(i, j, 1);
            rgb(i, j, 2) = rgba(i, j, 2);
        }
    }
    return 0;
}


int ns::transparent(OutTensor rgba, InTensor rgb, int alpha)
{
    rgb2rgba(rgba, rgb, alpha);
    int h = rgb.shape[HWC_H];
    int w = rgb.shape[HWC_W];
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            if (rgb(i, j, 0) == 255 &&
                    rgb(i, j, 1) == 255 &&
                    rgb(i, j, 2) == 255) {
                rgba(i, j, 3) = 0;
            }

        }
    }
    return 0;
}

int ns::normColor(OutTensor normRgb, InTensor rgb)
{
    int h = rgb.shape[HWC_H];
    int w = rgb.shape[HWC_W];
    normRgb = Tensor(rgb.shape);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float r = rgb(i, j, 0);
            float g = rgb(i, j, 1);
            float b = rgb(i, j, 2);
            float rho = std::sqrt(r*r + g*g + b*b);
            normRgb(i, j, 0) = 255*r/rho;
            normRgb(i, j, 1) = 255*g/rho;
            normRgb(i, j, 2) = 255*b/rho;
        }
    }
    return 0;
}

int ns::bayer2rgb(OutTensor rgb, InTensor xi)
{
    int padding = 1;
    Tensor bayer;
    ns::copyMakeBorder(bayer, xi, padding);
    int ho = bayer.shape[HWC_H];
    int wo = bayer.shape[HWC_W];
    rgb = Tensor(ho, wo, 3);
    for (int i = padding; i < ho - padding; i++) {
        for (int j = padding; j < wo - padding; j++) {
            /*
                |M00 M01 M02|
                |M10 M11 M12|
                |M20 M21 M22|

            */
            int n00 = (i - 1)*wo + j - 1;int n01 = (i - 1)*wo + j;int n02 = (i - 1)*wo + j + 1;
            int n10 = (i    )*wo + j - 1;int n11 = (i    )*wo + j;int n12 = (i    )*wo + j + 1;
            int n20 = (i + 1)*wo + j - 1;int n21 = (i + 1)*wo + j;int n22 = (i + 1)*wo + j + 1;
            if (i%2 == 0) {
                if (j%2 == 0) {
                    rgb(i, j, 0) = (bayer[n00] + bayer[n02] + bayer[n22] + bayer[n20])/4;
                    rgb(i, j, 1) = (bayer[n01] + bayer[n12] + bayer[n21] + bayer[n10])/4;
                    rgb(i, j, 2) = bayer[n11];
                } else {
                    rgb(i, j, 0) = (bayer[n01] + bayer[n21])/2;
                    rgb(i, j, 1) = bayer[n11];
                    rgb(i, j, 2) = (bayer[n10] + bayer[n12])/2;
                }
            } else {
                if (j%2 == 0) {
                    rgb(i, j, 0) = (bayer[n10] + bayer[n12])/2;
                    rgb(i, j, 1) = bayer[n11];
                    rgb(i, j, 2) = (bayer[n01] + bayer[n21])/2;
                } else {
                    rgb(i, j, 0) = bayer[n11];
                    rgb(i, j, 1) = (bayer[n01] + bayer[n12] + bayer[n21] + bayer[n10])/4;
                    rgb(i, j, 2) = (bayer[n00] + bayer[n02] + bayer[n22] + bayer[n20])/4;

                }
            }
        }
    }
    return 0;
}

Tensor ns::toTensor(int h, int w, int c, std::shared_ptr<uint8_t[]> &img)
{
    Tensor x(h, w, c);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        x[i] = img[i];
    }
    return x;
}

Tensor ns::load(const std::string &fileName)
{
    Tensor img;
    if (fileName.empty()) {
        return img;
    }
    std::shared_ptr<uint8_t[]> data = nullptr;
    int h = 0;
    int w = 0;
    int c = 0;
    if (fileName.find(".bmp") != std::string::npos) {
        c = 3;
        int ret = ns::BMP::load(fileName.c_str(), data, h, w);
        if (ret < 0) {
            return img;
        }
    } else if (fileName.find(".jpg") != std::string::npos) {
#ifdef ENABLE_JPEG
        int ret = imp::Jpeg::load(fileName.c_str(), data, h, w, c);
        if (ret < 0) {
            return img;
        }
#else
        std::cout<<"please enanle USE_JPEG."<<std::endl;
#endif
    }
    else if (fileName.find(".ppm") != std::string::npos) {
        c = 3;
        int ret = ns::PPM::load(fileName.c_str(), data, h, w);
        if (ret < 0) {
            return img;
        }
    } else {
        return img;
    }
    img = Tensor(h, w, c);
    for (std::size_t i = 0; i < img.totalSize; i++) {
        img.val[i] = data[i];
    }
    return img;
}

int ns::save(InTensor img, const std::string &fileName)
{
    if (fileName.empty()) {
        return -1;
    }
    int h = img.shape[HWC_H];
    int w = img.shape[HWC_W];
    int c = img.shape[HWC_C];
    /* rgb image */
    Tensor rgb;
    if (c == 1) {
        gray2rgb(rgb, img);
    } else if (c == 3) {
        rgb = img;
    } else if (c == 4) {
        rgba2rgb(rgb, img);
    } else {
        std::cout<<"unknown channel"<<std::endl;
        return -2;
    }

    /* clamp */
    std::shared_ptr<uint8_t[]> data = tensor2Image(rgb);
    /* save */
    if (fileName.find(".jpg") != std::string::npos) {
#ifdef ENABLE_JPEG
        int ret = imp::Jpeg::save(fileName.c_str(), data.get(), h, w, c);
        if (ret < 0) {
            return -3;
        }
#else
        std::cout<<"please enanle USE_JPEG."<<std::endl;
        return -4;
#endif
    } else if (fileName.find(".bmp") != std::string::npos) {
        int ret = ns::BMP::save(fileName, data, h, w);
        if (ret < 0) {
            return -3;
        }
    } else if (fileName.find(".ppm") != std::string::npos) {
        int ret = ns::PPM::save(fileName, data, h, w);
        if (ret < 0) {
            return -3;
        }
    } else {
        int ret = ns::BMP::save(fileName, data, h, w);
        if (ret < 0) {
            return -3;
        }
    }
    return 0;
}

int ns::resize(OutTensor xo, InTensor xi, const ns::Size &size, int type)
{
    switch (type) {
    case INTERPOLATE_NEAREST:
        ns::nearestInterpolate(xo, xi, size);
        break;
    case INTERPOLATE_BILINEAR:
        ns::bilinearInterpolate(xo, xi, size);
        break;
    case INTERPOLATE_CUBIC:
        ns::cubicInterpolate(xo, xi, size, cubic::bspLine);
        break;
    default:
        ns::nearestInterpolate(xo, xi, size);
        break;
    }
    return 0;
}

int ns::erode(OutTensor xo, InTensor xi, InTensor kernel, int maxIterateTimes)
{
    /*
        kernel:  0 --> ignore,
                 1 --> roi
    */
    int width = xi.shape[HWC_W];
    int height = xi.shape[HWC_H];
    int channel = xi.shape[HWC_C];
    int kernelSize = kernel.shape[HWC_H];
    int h = kernelSize/2;
    xo = xi;
    for (int it = 0; it < maxIterateTimes; it++) {
        Tensor xt = xo;
        for (int i = h; i < height - h; i++) {
            for (int j = h; j < width - h; j++) {
                for (int k = 0; k < channel; k++) {
                    float s = 0;
                    float minValue = 255;
                    for (int u = 0; u < kernelSize; u++) {
                        for (int v = 0; v < kernelSize; v++) {
                           float p = xt(i - h + u, j - h + v, k)*kernel(u, v);
                           if (p < minValue) {
                               minValue = p;
                           }
                           s += p;
                        }
                    }
                    if (s < 255) {
                        xo(i, j, k) = minValue;
                    }
                }
            }
        }
    }
    return 0;
}

int ns::dilate(OutTensor xo, InTensor xi, InTensor kernel, int maxIterateTimes)
{
    /*
        kernel:  0 --> ignore,
                 1 --> roi
    */
    int width = xi.shape[HWC_W];
    int height = xi.shape[HWC_H];
    int channel = xi.shape[HWC_C];
    int kernelSize = kernel.shape[HWC_H];
    int h = kernelSize/2;
    xo = xi;
    for (int it = 0; it < maxIterateTimes; it++) {
        Tensor xt = xo;
        for (int i = h; i < height - h; i++) {
            for (int j = h; j < width - h; j++) {
                for (int k = 0; k < channel; k++) {
                    float s = 0;
                    float maxValue = 0;
                    for (int u = 0; u < kernelSize; u++) {
                        for (int v = 0; v < kernelSize; v++) {
                            float p = xt(i - h + u, j - h + v, k)*kernel(u, v);
                            if (p > maxValue) {
                                maxValue = p;
                            }
                            s += p;
                        }
                    }

                    if (s > 255) {
                        xo(i, j, k) = maxValue;
                    }
                }
            }
        }
    }
    return 0;
}

int ns::traceBoundary(OutTensor xo, InTensor xi, std::vector<Point2i> &boundary)
{
    Tensor gray;
    rgb2gray(gray, xi);
    xo = Tensor(xi);
    int width = gray.shape[HWC_W];
    int height = gray.shape[HWC_H];
    /* boundary */
    for (int i = 0; i < height; i++) {
        gray(i, 0) = 255;
        gray(i, width - 1) = 255;
    }
    for (int i = 0; i < width; i++) {
        gray(0, i) = 255;
        gray(height - 1, i) = 255;;
    }

    Point2i startPoint;
    Point2i currentPoint;
    bool isAtStartPoint = true;
    int k = 0;
    Point2i d[8] = {{-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}};
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (gray(i, j) != 0) {
                continue;
            }
            startPoint = Point2i(i, j);
            currentPoint = startPoint;
            isAtStartPoint = true;
            while ((startPoint.x != currentPoint.x || startPoint.y != currentPoint.y) ||
                   isAtStartPoint) {
                isAtStartPoint = false;
                Point2i pos = currentPoint + d[k];
                int searchTime = 1;
                while (gray(pos.x, pos.y) == 255) {
                    k = (k + 1)%8;
                    pos = currentPoint + d[k];
                    if (++searchTime >= 8) {
                        pos = currentPoint;
                        break;
                    }
                }
                currentPoint = pos;
                boundary.push_back(pos);
                xo(currentPoint.x, currentPoint.y, 0) = 0;
                xo(currentPoint.x, currentPoint.y, 1) = 0;
                xo(currentPoint.x, currentPoint.y, 2) = 0;

                k -= 2;
                k = k < 0 ? (k + 8) : k;
            }
            break;
        }
    }
    return 0;
}

int ns::findConnectedRegion(OutTensor mask, InTensor xi, int connectCount, int &labelCount)
{
    Tensor gray(xi);
    Tensor kernel(3, 3);
    kernel.fill(1);
    int w = gray.shape[HWC_W];
    int h = gray.shape[HWC_H];
    /* boundary */
    for (int i = 0; i < h; i++) {
        gray(i, 0) = 255;
        gray(i, w - 1) = 255;
    }
    for (int i = 0; i < w; i++) {
        gray(0, i) = 255;
        gray(h - 1, i) = 255;;
    }
    Tensor SE({3, 3}, {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    });
    if (connectCount == 4) {
        SE(0, 0) = -1;
        SE(0, 2) = -1;
        SE(2, 0) = -1;
        SE(2, 2) = -1;
    }
    int label = 1;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            if (gray(i, j) != 0) {
                continue;
            }

            for (int u = 0; u < h; u++) {
                for (int v = 0; v < w; v++) {
                    if (mask(u, v) == 0) {
                        mask(u, v) = label;
                    }
                }
            }
            label++;
        }
    }
    return 0;
}

int ns::threshold(OutTensor xo, InTensor xi, float thres, float max_, float min_)
{
    if (xi.shape[HWC_C] != 1) {
        std::cout<<"invalid channel"<<std::endl;
        return -1;
    }
    xo = Tensor(xi.shape);
    for (std::size_t i = 0; i < xi.totalSize; i++) {
        if (xi.val[i] < thres) {
            xo.val[i] = min_;
        } else {
            xo.val[i] = max_;
        }
    }
    return 0;
}

int ns::detectThreshold(InTensor xi, int maxIter, int &thres, int &delta)
{
    if (xi.shape[HWC_C] != 1) {
        return -1;
    }
    int histogram[256] = {0};
    int maxVal = 0;
    int minVal = 255;
    /* histogram */
    for (std::size_t i = 0; i < xi.totalSize; i++) {
        int val = xi[i];
        if (val > maxVal) {
            maxVal = val;
        }
        if (val < minVal) {
            minVal = val;
        }
        histogram[val]++;
    }
    int newThreshold = (maxVal + minVal)/2;
    delta = maxVal - minVal;
    if (maxVal == minVal) {
        thres = newThreshold;
    } else {
        thres = 0;
        int t = maxIter;
        while (thres != newThreshold && t > 0) {
            thres = newThreshold;
            int totalGray = 0;
            int totalPixel = 0;
            for (int i = minVal; i < thres; i++) {
                totalGray += histogram[i]*i;
                totalPixel += histogram[i];
            }
            int mean1Gray = totalGray/totalPixel;
            totalGray = 0;
            totalPixel = 0;
            for (int i = thres; i <= maxVal; i++) {
                totalGray += histogram[i]*i;
                totalPixel += histogram[i];
            }
            int mean2Gray = totalGray/totalPixel;
            newThreshold = (mean1Gray + mean2Gray)/2;
            delta = std::abs(mean2Gray - mean1Gray);
            t--;
        }
    }
    return 0;
}

int ns::autoThreshold(OutTensor xo, InTensor xi, float max_, float min_)
{
    int thres = 0;
    int delta = 0;
    int ret = detectThreshold(xi, 100, thres, delta);
    if (ret != 0) {
        std::cout<<"detectThreshold failed"<<std::endl;
        return -1;
    }
    std::cout<<"auto thres:"<<thres<<std::endl;
    return threshold(xo, xi, thres, max_, min_);
}


int ns::otsuThreshold(OutTensor xo, InTensor xi, float max_, float min_)
{
    uint8_t thres = 0;
    int ret = otsu(xi, thres);
    std::cout<<"otsu thres:"<<thres<<std::endl;
    if (ret != 0) {
        std::cout<<"otsu failed"<<std::endl;
        return -1;
    }
    return threshold(xo, xi, thres, max_, min_);
}

int ns::entropyThreshold(OutTensor xo, InTensor xi, float max_, float min_)
{
    uint8_t thres = 0;
    int ret = entropy(xi, thres);
    std::cout<<"entropy thres:"<<thres<<std::endl;
    if (ret != 0) {
        return -1;
    }
    return threshold(xo, xi, thres, max_, min_);
}

int ns::regionGrow(OutTensor mask, InTensor xi, const Point2i &seed, const std::vector<uint8_t>& thres)
{
    if (xi.shape[HWC_C] != 1) {
        return -1;
    }
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    if (seed.x < 0 || seed.x > h ||
            seed.y < 0 || seed.y > w) {
        return -2;
    }
    float seedVal = xi(seed.x, seed.y);
    float s = seedVal;
    int total = 1;
    int count = 1;
    while (count > 0) {
        count = 0;
        for (int i = 1; i < h - 1; i++) {
            for (int j = 1; j < w - 1; j++) {

                for (int u = i - 1; u < i + 1; u++) {
                    for (int v = j - 1; v < j + 1; v++) {
                        if (mask(u, v) == 0) {
                            float delta = std::abs(xi(u, v) - seedVal);
                            if (delta <= thres[0]) {
                                mask(u, v) = 1;
                            }
                            for (std::size_t k = 1; k < thres.size(); k++) {
                                if (delta <= thres[k] && delta > thres[k - 1]) {
                                    mask(u, v) = k + 1;
                                }
                            }
                            if (mask(u, v) != 0) {
                                count++;
                                s += xi(u, v);
                            }
                        }
                    }
                }
            }
        }
        total += count;
        seedVal = s/total;
    }

    return 0;
}

int ns::templateMatch(InTensor xi, InTensor xt, Rect &rect)
{
    /*
        cosθ = <xi, xt>/(||xi||*||xt||)
    */
    Tensor grayi;
    if (xi.shape[HWC_C] != 1) {
        rgb2gray(grayi, xi);
    } else {
        grayi = xi;
    }
    Tensor grayt;
    if (xt.shape[HWC_C] != 1) {
        rgb2gray(grayt, xt);
    } else {
        grayt = xt;
    }
    rect.height = xt.shape[HWC_H];
    rect.width = xt.shape[HWC_W];
    float xtNorm = xt.norm2();
    int h = grayi.shape[HWC_H];
    int w = grayi.shape[HWC_W];
    /* find maximum cosθ */
    float maxCosTheta = 0;
    for (int i = 0; i < h - rect.height + 1; i++) {
        for (int j = 0; j < w - rect.width + 1; j++) {
#if 0
            Tensor x = grayi.block({i, j, 0}, {rect.height, rect.width, 1});
            float innerProduct = util::dot(x, grayt);
            float xNorm = x.norm2();
#else
            float innerProduct = 0;
            float xNorm = 0;
            for (int u = 0; u < rect.height; u++) {
                for (int v = 0; v < rect.width; v++) {
                    float x = grayi(i + u, j + v);
                    xNorm += x*x;
                    innerProduct += x*grayt(u, v);
                }
            }
            xNorm = std::sqrt(xNorm);
#endif
            float cosTheta = innerProduct/(xNorm*xtNorm);
            if (cosTheta > maxCosTheta) {
                rect.x = i;
                rect.y = j;
                maxCosTheta = cosTheta;
            }
        }
    }
    return 0;
}

int ns::houghLine(OutTensor xo, InTensor xi, float thres, int lineNo, const Color3 &color)
{
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    int c = xi.shape[HWC_C];
    /* binary image */
    Tensor xb;
    if (c != 1) {
        Tensor xg;
        rgb2gray(xg, xi);
        threshold(xb, xg, thres, 255, 0);
    } else {
        threshold(xb, xi, thres, 255, 0);
    }
    /* convert to polar coordinate  */
    int maxRho = std::sqrt(h*h + w*w);
    int maxAngle = 90;
    std::size_t areaNum = maxAngle*maxRho*2;
    /* area: (rho, theta) */
    Tensori transArea(2*maxRho, maxAngle);

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float p = xb(i, j, 0);
            if (p == 255) {
                for (int angle = 0; angle < maxAngle; angle++) {
                    float theta = 2*angle*pi/180.0;
                    int rho = float(j*std::cos(theta) + i*std::sin(theta));
                    if (rho >= 0) {
                        transArea(rho, angle) += 1;
                    } else {
                        rho = std::abs(rho);
                        transArea(maxRho + rho, angle) += 1;
                    }
                }
            }
        }
    }

    int maxRhoTolorance = 20;
    int maxAngleTolorance = 5;
    /* detect lines */
    std::vector<Hough::Line> lines(lineNo);
    Hough::Line maxValue;
    for (int k = 0; k < lineNo; k++) {
        maxValue.pixels = 0;
        for (std::size_t i = 0; i < areaNum; i++) {
            if (transArea[i] > maxValue.pixels) {
                maxValue.pixels = transArea[i];
                maxValue.angle = i;
            }
        }
        if (maxValue.pixels == 0) {
            return -1;
        }
        if (maxValue.angle < maxAngle*maxRho) {
            maxValue.rho = maxValue.angle/maxAngle;
            maxValue.angle = maxValue.angle%maxAngle;
        } else {
            maxValue.angle -= maxAngle*maxRho;
            maxValue.rho = -1*maxValue.angle/maxAngle;
            maxValue.angle = maxValue.angle%maxAngle;
        }
        lines[k].angle = 2*maxValue.angle;
        lines[k].rho = maxValue.rho;
        lines[k].pixels = maxValue.pixels;
        if (lines[k].rho < 0) {
            lines[k].angle -= 180;
            lines[k].rho *= -1;
        }

        for (int r = -1*maxRhoTolorance; r <= maxRhoTolorance; r++) {
            for (int a = -1*maxAngleTolorance; a <= maxAngleTolorance; a++) {
                int rho = maxValue.rho + r;
                int angle = 2*(maxValue.angle + a);
                if (angle < 0 && angle >= -180) {
                    angle += 180;
                    rho *= -1;
                }
                if (angle >= 180 && angle < 360) {
                    angle -= 180;
                    rho *= -1;
                }
                if (std::abs(rho) < maxRho &&
                        angle >= 0 && angle <= maxAngle*2) {
                    angle /= 2;
                    if (rho >= 0) {
                        transArea(rho, angle) = 0;
                    } else {
                        rho = std::abs(rho);
                        transArea(maxRho + rho, angle) = 0;
                    }
                }
            }
        }
    }

    /* draw line */
    xo = xi;
    for (int k = 0; k < lineNo; k++) {
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                float theta = lines[k].angle*pi/180.0;
                int rho = float(j*std::cos(theta) + i*std::sin(theta));
                if (rho == lines[k].rho) {
                    if (c != 1) {
                        xo(i, j, 0) = color.r;
                        xo(i, j, 1) = color.g;
                        xo(i, j, 2) = color.b;
                    } else {
                        xo(i, j, 0) = 255;
                    }
                }
            }
        }
    }
    return 0;
}

int ns::HOG(OutTensor xo, OutTensor hist, InTensor xi, int cellSize, int binSize, int blockSize)
{
    if (xi.shape[HWC_C] != 1) {
        return -1;
    }
    /* step1: align image */
    Tensor img;
    int h = (xi.shape[ns::HWC_H]/cellSize)*cellSize;
    int w = (xi.shape[ns::HWC_W]/cellSize)*cellSize;
    copy(img, xi, ns::Rect(0, 0, w, h));
    /* step2: compute grad and angle, keep shape */
    Tensor kx({3, 3}, {-1, 0, 1,
                       -2, 0, 2,
                       -1, 0, 1});
    Tensor ky({3, 3}, {-1, -2, -1,
                        0,  0,  0,
                        1,  2,  1});
    Tensor imgx;
    conv2d(imgx, kx, img, 1, 1);
    Tensor imgy;
    conv2d(imgy, ky, img, 1, 1);
    Tensor g = LinAlg::sqrt(imgx*imgx + imgy*imgy);
    Tensor theta(g.shape);
    for (std::size_t i = 0; i < theta.totalSize; i++) {
        theta[i] = std::atan2(imgy[i], imgx[i] + 1e-9)*180/pi;
    }

    /* step3: histogram of oriented gradient */
    int stride = cellSize;
    int angleUnit = 360/binSize;
    int hc = (h - cellSize)/stride + 1;
    int wc = (w - cellSize)/stride + 1;
    Tensor cellHist(hc, wc, binSize);
    for (int i = 0; i < hc; i++) {
        for (int j = 0; j < wc; j++) {
            for (int u = 0; u < cellSize; u++) {
                for (int v = 0; v < cellSize; v++) {
                    /* map to input */
                    int ui = u + i*stride;
                    int vj = v + j*stride;
                    if (ui >= h || vj >= w) {
                        continue;
                    }
                    /* cell histogram */
                    float angle = theta(ui, vj);
                    angle = angle < 0 ? angle + 180 : angle;
                    int index1 = angle/angleUnit;
                    int index2 = (index1 + 1)%binSize;
                    int mod = int(angle)%angleUnit;
                    float mag = g(ui, vj);
                    cellHist(i, j, index1) += mag*(1 - mod/angleUnit);
                    cellHist(i, j, index2) += mag*mod/angleUnit;
                }
            }
        }
    }
    /* step4: normalize */
    hist = Tensor(hc, wc, binSize);
    int hb = hc/blockSize;
    int wb = wc/blockSize;
    for (int i = 0; i < hb; i++) {
        for (int j = 0; j < wb; j++) {
            for (int k = 0; k < binSize; k++) {
                float s = 0;
                for (int h = 0; h < blockSize; h++) {
                    for (int l = 0; l < blockSize; l++) {
                        int u = h + i*cellSize;
                        int v = l + j*cellSize;
                        if (u >= hc || v >= wc) {
                            continue;
                        }
                        float mag = cellHist(u, v, k);
                        s += mag*mag;
                    }
                }
                s = std::sqrt(s);
                if (s == 0) {
                    continue;
                }
                for (int h = 0; h < blockSize; h++) {
                    for (int l = 0; l < blockSize; l++) {
                        int u = h + i*blockSize;
                        int v = l + j*blockSize;
                        if (u >= hc || v >= wc) {
                            continue;
                        }
                        cellHist(u, v, k) /= s;
                    }
                }
            }
        }
    }

    /* step5: hog image */
    xo = Tensor(h, w, 3);
    float maxMag = cellHist.max();
    for (int i = 0; i < hc; i++) {
        for (int j = 0; j < wc; j++) {
            float angle = 0;
            for (int k = 0; k < binSize; k++) {
                angle = angle > 180 ? 180 - angle : angle;
                float rad = angle*pi/180;
                float sinTheta = std::sin(rad);
                float cosTheta = std::cos(rad);
                float mag = cellHist(i, j, k)/maxMag;
                int x1 = float(i*cellSize + mag*cellSize*cosTheta*0.5);
                int y1 = float(j*cellSize + mag*cellSize*sinTheta*0.5);
                int x2 = float(i*cellSize - mag*cellSize*cosTheta*0.5);
                int y2 = float(j*cellSize - mag*cellSize*sinTheta*0.5);
                line(xo, Point2i(y1, x1), Point2i(y2, x2), Color3(0, 255*std::sqrt(mag), 0));
                angle += angleUnit;
            }
        }
    }
    return 0;
}

int ns::harrisCorner(OutTensor xo, InTensor xi, float coeff)
{
    if (xi.shape[HWC_C] != 1) {
        return -1;
    }
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    xo = Tensor(xi.shape);
    /* step1: gradient */
    Tensor kx({3, 3}, {-1, 0, 1,
                       -2, 0, 2,
                       -1, 0, 1});
    Tensor ky({3, 3}, {-1, -2, -1,
                        0,  0,  0,
                        1,  2,  1});
    Tensor gx;
    conv2d(gx, kx, xi, 1, 1);
    Tensor gy;
    conv2d(gy, ky, xi, 1, 1);

    Tensor gxx = gx*gx;
    Tensor gyy = gy*gy;
    Tensor gxy = gx*gy;
    /* step2: sum */
    Tensor ks = Tensor::ones(2, 2);
    Tensor fxx;
    conv2d(fxx, ks, gxx, 1, 0);
    Tensor fyy;
    conv2d(fyy, ks, gyy, 1, 0);
    Tensor fxy;
    conv2d(fxy, ks, gxy, 1, 0);
    /* step3: */
    for (int i = 1; i < h - 1; i++) {
        for (int j = 1; j < w - 1; j++) {
            float sxx = fxx(i, j, 0);
            float syy = fyy(i, j, 0);
            float sxy = fxy(i, j, 0);
            /*
              M = |sxx sxy|
                  |sxy syy|
            */
            float det = sxx*syy - sxy*sxy;
            float trace = sxx + syy;
            float resp = det - coeff*trace*trace;
            xo(i, j, 0) = resp;
        }
    }
    return 0;
}

int ns::hsvHistogramEqualize(OutTensor xo, InTensor xi)
{
    Tensor hsv;
    ns::RGB2HSV(hsv, xi);
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    Tensor v(h, w, 1);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            v(i, j) = hsv(i, j, 2);
        }
    }
    Tensor veq;
    ns::histogramEqualize(veq, v);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            hsv(i, j, 2) = veq(i, j);
        }
    }
    ns::HSV2RGB(xo, hsv);
    return 0;
}

int ns::floydSteinbergDithering(OutTensor xo, InTensor xi)
{
    if (xi.shape[HWC_C] != 1) {
        return -1;
    }
    int h = xi.shape[0];
    int w = xi.shape[1];
    xo = Tensor(h, w, 1);
    for (int i = 1; i < h - 1; i++) {
        for (int j = 1; j < w - 1; j++) {
            int p1 = xi(i, j);
            float p2 = p1/255*255;
            float e = p2 - p1;
            xo(i    , j + 1) = p2 + e*7/16;
            xo(i + 1, j - 1) = p2 + e*3/16;
            xo(i + 1, j)     = p2 + e*5/16;
            xo(i + 1, j + 1) = p2 + e*1/16;
        }
    }
    return 0;
}

int ns::atkinsonDithering(OutTensor xo, InTensor xi)
{
    if (xi.shape[HWC_C] != 1) {
        return -1;
    }
    int h = xi.shape[0];
    int w = xi.shape[1];
    xo = Tensor(xi);
    Tensor error(h, w);
    for (int i = 1; i < h - 2; i++) {
        for (int j = 1; j < w - 2; j++) {
            int p1 = xo(i, j) + error(i, j);
            float p2 = p1/255*255;
            xo(i, j) = p2;
            float e = (p2 - p1)/8.0;
            error(i    , j + 1) += e;
            error(i    , j + 2) += e;
            error(i + 1, j - 1) += e;
            error(i + 1, j)     += e;
            error(i + 1, j + 1) += e;
            error(i + 2, j)     += e;
        }
    }
    return 0;
}
