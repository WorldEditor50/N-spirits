#include "mat.h"

std::default_random_engine Mat::engine;

Mat Mat::zeros(std::size_t rows, std::size_t cols)
{
    return Mat(rows, cols);
}

Mat Mat::identity(std::size_t rows, std::size_t cols)
{
    Mat y(rows, cols);
    for (std::size_t i = 0; i < y.rows; i++) {
        for (std::size_t j = 0; j < y.cols; j++) {
            if (i == j) {
                y.data[i*rows + j] = 1;
            }
        }
    }
    return y;
}

Mat &Mat::operator=(const Mat &r)
{
    if (this == &r) {
        return *this;
    }
    rows = r.rows;
    cols = r.cols;
    data = std::vector<float>(r.data);
    return *this;
}

Mat &Mat::operator=(Mat &&r)
{
    if (this == &r) {
        return *this;
    }
    rows = r.rows;
    cols = r.cols;
    data.swap(r.data);
    r.rows = 0;
    r.cols = 0;
    return *this;
}

Mat Mat::operator *(const Mat &x) const
{
    if (cols != x.rows) {
        return Mat();
    }
    Mat y(rows, x.cols);
    for (std::size_t i = 0; i < y.rows; i++) {
        for (std::size_t j = 0; j < y.cols; j++) {
            for (std::size_t k = 0; k < cols; k++) {
                y.data[i*y.rows + j] += data[i*rows + k] * x.data[k*x.rows + j];
            }
        }
    }
    return y;
}

Mat Mat::operator /(const Mat &x) const
{
    Mat y(rows, cols);
    for (std::size_t i = 0; i < y.data.size(); i++) {
        y.data[i] = data[i]/x.data[i];
    }
    return y;
}

Mat Mat::operator +(const Mat &x) const
{
    Mat y(rows, cols);
    for (std::size_t i = 0; i < y.data.size(); i++) {
        y.data[i] = data[i] + x.data[i];
    }
    return y;
}

Mat Mat::operator -(const Mat &x) const
{
    Mat y(rows, cols);
    for (std::size_t i = 0; i < y.data.size(); i++) {
        y.data[i] = data[i] - x.data[i];
    }
    return y;
}

Mat Mat::operator %(const Mat &x) const
{
    Mat y(rows, cols);
    for (std::size_t i = 0; i < y.data.size(); i++) {
        y.data[i] = data[i] * x.data[i];
    }
    return y;
}

Mat &Mat::operator /=(const Mat &x)
{
    for (std::size_t i = 0; i < data.size(); i++) {
        data[i] /= x.data[i];
    }
    return *this;
}

Mat &Mat::operator +=(const Mat &x)
{
    for (std::size_t i = 0; i < data.size(); i++) {
        data[i] += x.data[i];
    }
    return *this;
}

Mat &Mat::operator -=(const Mat &x)
{
    for (std::size_t i = 0; i < data.size(); i++) {
        data[i] -= x.data[i];
    }
    return *this;
}

Mat &Mat::operator %=(const Mat &x)
{
    for (std::size_t i = 0; i < data.size(); i++) {
        data[i] *= x.data[i];
    }
    return *this;
}

Mat Mat::operator *(float x) const
{
    Mat y(rows, cols);
    for (std::size_t i = 0; i < data.size(); i++) {
        y.data[i] = data[i] * x;
    }
    return y;
}

Mat Mat::operator /(float x) const
{
    Mat y(rows, cols);
    for (std::size_t i = 0; i < data.size(); i++) {
        y.data[i] = data[i] / x;
    }
    return y;
}

Mat Mat::operator +(float x) const
{
    Mat y(rows, cols);
    for (std::size_t i = 0; i < data.size(); i++) {
        y.data[i] = data[i] + x;
    }
    return y;
}

Mat Mat::operator -(float x) const
{
    Mat y(rows, cols);
    for (std::size_t i = 0; i < data.size(); i++) {
        y.data[i] = data[i] - x;
    }
    return y;
}

Mat &Mat::operator *=(float x)
{
    for (std::size_t i = 0; i < data.size(); i++) {
        data[i] *= x;
    }
    return *this;
}

Mat &Mat::operator /=(float x)
{
    for (std::size_t i = 0; i < data.size(); i++) {
        data[i] /= x;
    }
    return *this;
}

Mat &Mat::operator +=(float x)
{
    for (std::size_t i = 0; i < data.size(); i++) {
        data[i] += x;
    }
    return *this;
}

Mat &Mat::operator -=(float x)
{
    for (std::size_t i = 0; i < data.size(); i++) {
        data[i] -= x;
    }
    return *this;
}

Mat Mat::tr() const
{
    Mat y(cols, rows);
    for (std::size_t i = 0; i < y.rows; i++) {
        for (std::size_t j = 0; j < y.cols; j++) {
            y.data[i*y.rows + j] = data[j*rows + i];
        }
    }
    return y;
}

Mat Mat::sub(std::size_t pr, std::size_t pc, std::size_t sr, std::size_t sc) const
{
    Mat y(sr, sc);
    for (std::size_t i = 0; i < y.rows; i++) {
        for (std::size_t j = 0; j < y.cols; j++) {
            y.data[i*y.rows + j] = data[(i + pr)*rows + j + pc];
        }
    }
    return y;
}

Mat Mat::flatten() const
{
    Mat y(rows*cols, 1, data);
    return y;
}

Mat Mat::f(std::function<float(float)> func) const
{
    Mat y(rows, cols);
    for (std::size_t i = 0; i < data.size(); i++) {
        y.data[i] = func(data[i]);
    }
    return y;
}

void Mat::set(std::size_t pr, std::size_t pc, const Mat &x)
{
    for (std::size_t i = pr; i < pr + x.rows; i++) {
        for (std::size_t j = pc; j < pc + x.cols; j++) {
            data[i*rows + j] = x.data[(i - pr)*rows + j - pc];
        }
    }
    return;
}

void Mat::zero()
{
    data.assign(totalSize, 0);
    return;
}

void Mat::full(float x)
{
    data.assign(totalSize, x);
    return;
}

void Mat::EMA(const Mat &r, float rho)
{
    for (std::size_t i = 0; i < data.size(); i++) {
        data[i] = (1 - rho) * data[i] + rho * r.data[i];
    }
    return;
}

void Mat::show() const
{
    std::cout<<"row:"<<rows<<", cols"<<cols<<std::endl;
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            std::cout<<data[i*rows + j]<<" ";
        }
        std::cout<<std::endl;
    }
    return;
}

int Mat::mul(Mat &y, const Mat &x1, const Mat &x2)
{
    if (x1.cols != x2.rows) {
        return -1;
    }
    if ((y.rows != x1.rows) && (y.cols != x2.cols)) {
        return -2;
    }
    for (std::size_t i = 0; i < y.rows; i++) {
        for (std::size_t j = 0; j < y.cols; j++) {
            for (std::size_t k = 0; k < x1.cols; k++) {
                y.data[i*y.rows + j] += x1.data[i*x1.rows + k]*x2.data[k*x2.rows + j];
            }
        }
    }
    return 0;
}

float Mat::dot(const Mat &x1, const Mat &x2)
{
    float s = 0;
    for (std::size_t i = 0; i < x1.totalSize; i++) {
        s += x1.data[i] * x2.data[i];
    }
    return s;;
}

void Mat::div(Mat &y, const Mat &x1, const Mat &x2)
{
    for (std::size_t i = 0; i < x1.totalSize; i++) {
        y.data[i] = x1.data[i] / x2.data[i];
    }
    return;
}

void Mat::add(Mat &y, const Mat &x1, const Mat &x2)
{
    for (std::size_t i = 0; i < x1.totalSize; i++) {
        y.data[i] = x1.data[i] + x2.data[i];
    }
    return;
}

void Mat::minus(Mat &y, const Mat &x1, const Mat &x2)
{
    for (std::size_t i = 0; i < x1.totalSize; i++) {
        y.data[i] = x1.data[i] - x2.data[i];
    }
    return;
}

Mat Mat::kronecker(const Mat &x1, const Mat &x2)
{
    Mat y(x1.rows * x2.rows, x1.cols * x2.cols);
    for (std::size_t i = 0; i < x1.rows; i++) {
        for (std::size_t j = 0; j < x1.cols; j++) {
            for (std::size_t k = 0; k < x2.rows; k++) {
                for (std::size_t l = 0; l < x2.cols; l++) {
                    y(i*x2.rows + k, j*x2.cols + l) = x1.data[i*x1.rows + j] * x2.data[k*x2.rows + l];
                }
            }
        }
    }
    return y;
}

Mat Mat::sqrt(const Mat &x)
{
    Mat y(x.rows, x.cols);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.data[i] = std::sqrt(x.data[i]);
    }
    return y;
}

Mat Mat::exp(const Mat &x)
{
    Mat y(x.rows, x.cols);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.data[i] = std::exp(x.data[i]);
    }
    return y;
}

Mat Mat::sin(const Mat &x)
{
    Mat y(x.rows, x.cols);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.data[i] = std::sin(x.data[i]);
    }
    return y;
}

Mat Mat::cos(const Mat &x)
{
    Mat y(x.rows, x.cols);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.data[i] = std::cos(x.data[i]);
    }
    return y;
}

void Mat::uniform(Mat &x)
{
    std::uniform_real_distribution<float> distibution(-1, 1);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        x.data[i] = distibution(engine);
    }
    return;
}

void Mat::diag(const Mat &x, Mat &elements, int k)
{

    return;
}

int Mat::det(const Mat &x, float &value)
{
    if (x.rows != x.cols) {
        return -1;
    }
    /* 1-order */
    if (x.rows == 1) {
        return x(0, 0);
    }
    /* 2-order */
    if (x.rows == 2) {
        return x(0, 0)*x(1, 1) - x(0, 1)*x(1, 0);
    }

    /*
       n-order:
        Gaussian Elimination -> up triangle matrix -> det(x) = ∏ Diag(U)
    */

    Mat d(x);
    std::size_t N = x.cols;
    value = 1;
    for (std::size_t i = 0; i < N; i++) {
        /* find 0 in column */
        std::size_t t = i;
        while (d(t, i) == 0 && t < N) {
            t++;
        }
        if (t == N) {
            continue;
        }
        for (std::size_t j = 0; j < N; j++) {
            float tmp = d(i, j);
            d(i, j) = d(t, j);
            d(t, j) = tmp;
        }

        if (t != i) {
            value *= -1;
        }
        /* ∏ Diag(U)  */
        float diagonal = d(i, i);
        value *= diagonal;
        for (std::size_t j = 0; j < N; j++) {
            d(i, j) /= diagonal;
        }

        for (std::size_t j = 0; j < N; j++) {
            if (i == j) {
                continue;
            }
            float dji = d(j, i);
            for (std::size_t k = 0; k < N; k++) {
                d(j, k) -= dji*d(i, k);
            }
        }
    }
    return 0;
}

Mat Mat::tanh(const Mat &x)
{
    Mat y(x.rows, x.cols);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.data[i] = std::tanh(x.data[i]);
    }
    return y;
}

float Mat::LU::sumLrkUki(Mat &l, std::size_t r, Mat &u, std::size_t i)
{
    float re = 0.0;
    for (std::size_t k = 0; k < r; k++) {
        re += l(r, k) * u(k, i);
    }
    return re;
}
float Mat::LU::sumLikUkr(Mat &u, std::size_t r, Mat &l, std::size_t i)
{
    float re = 0.0;
    for (std::size_t k = 0; k < r; k++) {
        re += l(i, k) * u(k, r);
    }
    return re;
}

int Mat::LU::solve(const Mat &x, Mat &l, Mat &u)
{
    /* square matrix */
    if (x.rows != x.cols) {
        return -1;
    }
    /* U1i=X1i,Li1=Xi1/U11 */
    for (std::size_t i = 0; i < x.rows; i++) {
         u(0, i) = x(0, i);
         l(i, 0) = x(i, 0)/u(0, 0);
     }
     /*
        Uri= Xri- Σ(k=1->r-1)Lrk*Uki (i=r,r+1,...,n)
        Lir= Xir - Σ(k=1->r-1)Lik*Ukr (i=r+1,r+2,...,n and r≠n)
    */
     std::size_t N = x.rows - 1;
     for (std::size_t r = 1; r < x.rows; r++) {
         for (std::size_t i = r; i < x.cols; i++) {
             u(r, i) = x(r, i) - sumLrkUki(l, r, u, i);
             if (i == r) {
                 l(r, r) = 1;
             } else if (r == N) {
                 l(N, N) = 1;
             } else {
                 l(i, r) = (x(i, r) - sumLikUkr(l, i, u, r))/u(r, r);
             }
         }
    }
    return 0;
}

int Mat::LU::inv(const Mat &x, Mat &xi)
{
    Mat l(x.rows, x.cols);
    Mat u(x.rows, x.cols);
    /* LU decomposition */
    int ret = LU::solve(x, l, u);
    if (ret < 0) {
        return -1;
    }
    /* check invertable */
    float prod = 1;
    for (std::size_t i = 1; i < x.rows; i++) {
        prod *= u(i, i);
    }
    if (prod == 0) {
        return -2;
    }

    /* inverse matrix of u */
    Mat ui(x.rows, x.cols);
    for (std::size_t i = 0; i < ui.rows; i++) {
        ui(i, i) = 1/u(i, i);
        for (int k = i - 1; k >= 0; k--) {
            float s = 0;
            for (std::size_t j = k + 1; j <= i; j++) {
                s += u(k, j)*ui(j, i);
            }
            ui(k, i) = -s/u(k, k);
        }
    }
    /* inverse matrix of l */
    Mat li(x.rows, x.cols);
    for (std::size_t i = 0; i < li.cols; i++) {
        li(i, i) = 1;
        for (std::size_t k = i + 1; k < x.rows; k++) {
            for (std::size_t j = i; j <= k - 1; j++) {
                li(k, i) -= l(k, j)*li(j, i);
            }
        }
    }
    /* inverse matrix of x, x = l*u , xi = ui*li */
    xi = Mat(x.rows, x.cols);
    Mat::mul(xi, ui, li);
    return 0;
}

void Mat::parse(std::istringstream &stream, std::size_t cols, Mat &x)
{
    /* csv:<rows><cols><data> */
    std::size_t row = 0;
    std::size_t col = 0;
    /* rows */
    std::string rowData;
    std::getline(stream, rowData, ',');
    row = std::atoi(rowData.c_str());
    /* cols */
    std::string colData;
    std::getline(stream, colData, ',');
    col = std::atoi(colData.c_str());
    /* data */
    x = Mat(row, col);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        std::string data;
        std::getline(stream, data, ',');
        x.data[i] = std::atof(data.c_str());
    }
    return;
}

void Mat::toString(const Mat &x, std::string &line)
{
    /* csv:<rows><cols><data> */
    line += std::to_string(x.rows) + "," + std::to_string(x.cols);
    for (std::size_t i = 0; i < x.rows; i++) {
        for (std::size_t j = 0; j < x.cols; j++) {
            line += std::to_string(x(i, j));
            if (j < x.cols - 1) {
                line += ",";
            }
        }
    }
    return;
}
