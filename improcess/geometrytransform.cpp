#include "geometrytransform.h"

int improcess::move(const Tensor &x, const Point2i &offset, Tensor &y)
{
    if (offset.x < 0 || offset.x > x.shape[HWC_W] ||
        offset.y < 0 || offset.y > x.shape[HWC_H]) {
        return -1;
    }
    for (int i = 0; i < x.shape[HWC_H]; i++) {
        for (int j = 0; j < x.shape[HWC_W]; j++) {
            if (i < offset.y || j < offset.x) {
                continue;
            }
            y(i, j, 0) = x(i - offset.y, j - offset.x, 0);
            y(i, j, 1) = x(i - offset.y, j - offset.x, 1);
            y(i, j, 2) = x(i - offset.y, j - offset.x, 2);
        }
    }
    return 0;
}

int improcess::transpose(const Tensor &x, Tensor &y)
{
    /* HWC -> WHC */
    y = x.permute(1, 0, 2);
    return 0;
}
