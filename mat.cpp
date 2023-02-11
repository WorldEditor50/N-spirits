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
                y.data[i*cols + j] = 1;
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
    totalSize = r.totalSize;
    if (data.empty()) {
        data = std::vector<float>(r.data);
    } else {
        data.assign(r.data.begin(), r.data.end());
    }
    return *this;
}

Mat &Mat::operator=(Mat &&r)
{
    if (this == &r) {
        return *this;
    }
    rows = r.rows;
    cols = r.cols;
    totalSize = r.totalSize;
    data.swap(r.data);
    r.rows = 0;
    r.cols = 0;
    r.totalSize = 0;
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
                y.data[i*y.cols + j] += data[i*cols + k] * x.data[k*x.cols + j];
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
            y.data[i*y.cols + j] = data[j*cols + i];
        }
    }
    return y;
}

int Mat::reshape(size_t rows_, size_t cols_)
{
    if (rows_*cols_ != totalSize) {
        return -1;
    }
    rows = rows_;
    cols = cols_;
    return 0;
}

Mat Mat::sub(std::size_t pr, std::size_t pc, std::size_t sr, std::size_t sc) const
{
    Mat y(sr, sc);
    for (std::size_t i = 0; i < y.rows; i++) {
        for (std::size_t j = 0; j < y.cols; j++) {
            y.data[i*y.cols + j] = data[(i + pr)*cols + j + pc];
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
            data[i*cols + j] = x.data[(i - pr)*cols + j - pc];
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
    std::cout<<"row:"<<rows<<", cols:"<<cols<<std::endl;
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            std::size_t index = i*cols + j;
            std::cout<<data[index]<<" ";
        }
        std::cout<<std::endl;
    }
    return;
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
                    y(i*x2.cols + k, j*x2.cols + l) = x1.data[i*x1.cols + j] * x2.data[k*x2.cols + l];
                }
            }
        }
    }
    return y;
}

void Mat::Swap::row(Mat &x, size_t ri, size_t rj)
{
    for (std::size_t h = 0; h < x.cols; h++) {
        float tmp = x(ri, h);
        x(ri, h) = x(rj, h);
        x(rj, h) = tmp;
    }
    return;
}

void Mat::Swap::col(Mat &x, size_t ci, size_t cj)
{
    for (std::size_t h = 0; h < x.rows; h++) {
        float tmp = x(h, ci);
        x(h, ci) = x(h, cj);
        x(h, cj) = tmp;
    }
    return;
}

void Mat::Multiply::ikkj(Mat &y, const Mat &x1, const Mat &x2)
{
    /* no transpose */
    for (std::size_t i = 0; i < y.rows; i++) {
        for (std::size_t j = 0; j < y.cols; j++) {
            for (std::size_t k = 0; k < x1.cols; k++) {
                /* (i, j) = (i, k) * (k, j) */
                y.data[i*y.cols + j] += x1.data[i*x1.cols + k]*x2.data[k*x2.cols + j];
            }
        }
    }
    return;
}

void Mat::Multiply::ikjk(Mat &y, const Mat &x1, const Mat &x2)
{
    /* transpose x2 */
    for (std::size_t i = 0; i < y.rows; i++) {
        for (std::size_t j = 0; j < y.cols; j++) {
            for (std::size_t k = 0; k < x1.cols; k++) {
                /* (i, j) = (i, k) * (j, k)^T */
                y.data[i*y.cols + j] += x1.data[i*x1.cols + k]*x2.data[j*x2.cols + k];
            }
        }
    }
    return;
}

void Mat::Multiply::kikj(Mat &y, const Mat &x1, const Mat &x2)
{
    /* transpose x1 */
    for (std::size_t i = 0; i < y.rows; i++) {
        for (std::size_t j = 0; j < y.cols; j++) {
            for (std::size_t k = 0; k < x1.rows; k++) {
                /* (i, j) = (k, i)^T * (k, j)^T */
                y.data[i*y.cols + j] += x1.data[k*x1.cols + i]*x2.data[k*x2.cols + j];
            }
        }
    }
    return;
}

void Mat::Multiply::kijk(Mat &y, const Mat &x1, const Mat &x2)
{
    /* transpose x1, x2 */
    for (std::size_t i = 0; i < y.rows; i++) {
        for (std::size_t j = 0; j < y.cols; j++) {
            for (std::size_t k = 0; k < x1.rows; k++) {
                /* (i, j) = (k, i)^T * (j, k)^T */
                y.data[i*y.cols + j] += x1.data[k*x1.cols + i] * x2.data[j*x2.cols + k];
            }
        }
    }
    return;
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
    line += std::to_string(x.rows) + "," + std::to_string(x.cols) + ",";
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


