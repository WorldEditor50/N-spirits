#include "color.h"

int imp::RGB2CMY(OutTensor xo, InTensor xi)
{
    if (xi.shape[HWC_C] != 3) {
        return -1;
    }
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    xo = Tensor(h, w, 3);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            xo(i, j, 0) = 255 - xi(i, j, 0);
            xo(i, j, 1) = 255 - xi(i, j, 1);
            xo(i, j, 2) = 255 - xi(i, j, 2);
        }
    }
    return 0;
}

int imp::RGB2HSI(OutTensor xo, InTensor xi)
{
    if (xi.shape[HWC_C] != 3) {
        return -1;
    }
    /*
        H = theta (b <= g)
        H = 360 - theta

        theta = arcos(((r - g) + (r - b))/2/(sqrt((r -g)^2 + (r -b)^2)))

        S = 1 - 3*minRGB/(r + g + b)

        I = (r + g + b)/3
    */
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    xo = Tensor(h, w, 3);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float r = xi(i, j, 0)/255.0;
            float g = xi(i, j, 1)/255.0;
            float b = xi(i, j, 2)/255.0;
            float maxRGB = std::max(std::max(r, g), b);
            float minRGB = std::min(std::min(r, g), b);
            float H, S, I;

            I = (r + g + b)/3.0;
            if (I < 0.078431 || I > 0.92) {
                S = 0;
            } else {
                S = 1 - minRGB/I;
            }
            if (maxRGB == minRGB) {
                H = 0;
                S = 0;
            }
            float rad;
            float Q = ((r -g) + (r - b))/2.0/(std::sqrt((r - g)*(r - g) + (r - b)*(r - b)));
            if (Q > (1 - 1e-9)) {
                rad = 0;
            } else if (Q < (1 - 1e-9)) {
                rad = pi;
            } else {
                rad = std::acos(Q);
            }
            float theta = rad*180/pi;
            if (b > g) {
                H = 360 - theta;
            } else {
                H = theta;
            }

            I *= 255;
            S *= 255;
            xo(i, j, 0) = H;
            xo(i, j, 1) = S;
            xo(i, j, 2) = I;
        }
    }
    return 0;
}

int imp::HSI2RGB(OutTensor xo, InTensor xi)
{
    if (xi.shape[HWC_C] != 3) {
        return -1;
    }
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    xo = Tensor(xi.shape);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float H = xi(i, j, 0);
            float S = xi(i, j, 1)/255.0f;
            float I = xi(i, j, 2)/255.0f;
            float R = 0;
            float G = 0;
            float B = 0;
            if (H >= 0 && H < 120) {
                B = I*(1 - S);
                R = I*(1 + S*std::cos(H)/std::cos(60 - H));
                G = 3*I - B - R;
            } else if (H > 120 && H < 240) {
                H = H - 120;
                R = I*(1 - S);
                G = I*(1 + S*std::cos(H)/std::cos(60 - H));
                B = 3*I - R - G;
            } else {
                H = H - 240;
                G = I*(1 - S);
                B = I*(1 + S*std::cos(H)/std::cos(60 - H));
                R = 3*I - B - G;
            }
            xo(i, j, 0) = R;
            xo(i, j, 1) = G;
            xo(i, j, 2) = B;
        }
    }
    return 0;
}

int imp::RGB2HSV(OutTensor xo, InTensor xi)
{
    if (xi.shape[HWC_C] != 3) {
        return -1;
    }
    /*
        H = (G - B)/(MAX - MIN)*60, R = MAX
        H = 2 + (B - R)/(MAX - MIN)*60, G = MAX
        H = 4 + (R - G)/(MAX - MIN)*60, B = MAX
        S = (MAX - MIN)/MAX
        V = MAX
    */
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    xo = Tensor(xi.shape);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float R = xi(i, j, 0)/255.0;
            float G = xi(i, j, 1)/255.0;
            float B = xi(i, j, 2)/255.0;
            float maxVal = std::max(std::max(R, G), B);
            float minVal = std::min(std::min(R, G), B);
            float H = 0;
            float S = 0;
            float V = maxVal;
            float delta = maxVal - minVal;
            if (maxVal != 0) {
                S = delta/maxVal;
            } else {
                return -1;
            }
            if (maxVal == R) {
                H = (G - B)/delta;
            } else if (maxVal == G) {
                H = 2 + (B - R)/delta;
            } else {
                H = 4 + (R - G)/delta;
            }
            H *= 60;
            H = H < 0 ? (H + 360) : H;
            H /= 360;
            H *= 255;
            S *= 255;
            xo(i, j, 0) = H;
            xo(i, j, 1) = S;
            xo(i, j, 2) = V;
        }
    }
    return 0;
}

int imp::HSV2RGB(OutTensor xo, InTensor xi)
{
    if (xi.shape[HWC_C] != 3) {
        return -1;
    }
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    xo = Tensor(xi.shape);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float H = xi(i, j, 0)/255.0f*360;
            float S = xi(i, j, 1)/255.0f;
            float V = xi(i, j, 2)/255.0f;
            float R = 0;
            float G = 0;
            float B = 0;
            if (S == 0) {
                R = V;
                G = V;
                B = V;
            }
            int n = std::floor(H/60);
            float tmp = H/60;
            float f = tmp - n;
            float p = V*(1 - S);
            float q = V*(1 - f*S);
            float t = V*(1 - (1 - f)*S);
            switch (n) {
            case 0:
                R = V;
                G = t;
                B = p;
                break;
            case 1:
                R = q;
                G = V;
                B = p;
                break;
            case 2:
                R = p;
                G = V;
                B = t;
                break;
            case 3:
                R = p;
                G = q;
                B = V;
                break;
            case 4:
                R = t;
                G = p;
                B = q;
                break;
            default:
                R = V;
                G = p;
                B = q;
                break;
            }
            xo(i, j, 0) = R;
            xo(i, j, 1) = G;
            xo(i, j, 2) = B;
        }
    }
    xo *= 255;
    return 0;
}

int imp::RGB2YUV(OutTensor xo, InTensor xi)
{
    if (xi.shape[HWC_C] != 3) {
        return -1;
    }
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    xo = Tensor(xi.shape);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float R = xi(i, j, 0);
            float G = xi(i, j, 1);
            float B = xi(i, j, 2);
            float Y = 0.299*R + 0.587*G + 0.114*B;
            float U = (B - Y)*0.567;
            float V = (R - Y)*0.713;
            xo(i, j, 0) = bound(Y, 0, 255);
            xo(i, j, 1) = bound(U, 0, 255);
            xo(i, j, 2) = bound(V, 0, 255);
        }
    }
    return 0;
}

int imp::YUV2RGB(OutTensor xo, InTensor xi)
{
    if (xi.shape[HWC_C] != 3) {
        return -1;
    }
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    xo = Tensor(xi.shape);
    /*
        1.0, 0.0, 1.402,
        1.0, -0.344, -0.714,
        1.0, 1.772, 0
    */
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float Y = xi(i, j, 0);
            float U = xi(i, j, 1);
            float V = xi(i, j, 2);
            float R = Y + 1.402*V;
            float G = Y - 0.344*U - 0.714*V;
            float B = Y + 1.772*U;
            xo(i, j, 0) = bound(R, 0, 255);
            xo(i, j, 1) = bound(G, 0, 255);
            xo(i, j, 2) = bound(B, 0, 255);
        }
    }
    return 0;
}

int imp::RGB2YIQ(OutTensor xo, InTensor xi)
{
    if (xi.shape[HWC_C] != 3) {
        return -1;
    }
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    xo = Tensor(xi.shape);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float R = xi(i, j, 0);
            float G = xi(i, j, 1);
            float B = xi(i, j, 2);
            float Y = 0.299*R + 0.587*G + 0.114*B;
            float I = 0.596*R - 0.274*G - 0.322*B;
            float Q = 0.211*R - 0.523*G + 0.312*B;
            xo(i, j, 0) = bound(Y, 0, 255);
            xo(i, j, 1) = bound(I, 0, 255);
            xo(i, j, 2) = bound(Q, 0, 255);
        }
    }
    return 0;
}

int imp::YIQ2RGB(OutTensor xo, InTensor xi)
{
    if (xi.shape[HWC_C] != 3) {
        return -1;
    }
    int h = xi.shape[HWC_H];
    int w = xi.shape[HWC_W];
    xo = Tensor(xi.shape);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float Y = xi(i, j, 0);
            float I = xi(i, j, 1);
            float Q = xi(i, j, 2);
            float R = Y + 0.956*I + 0.621*Q;
            float G = Y - 0.262*I - 0.647*Q;
            float B = Y - 1.106*I + 1.703*Q;
            xo(i, j, 0) = bound(R, 0, 255);
            xo(i, j, 1) = bound(G, 0, 255);
            xo(i, j, 2) = bound(B, 0, 255);
        }
    }
    return 0;
}
