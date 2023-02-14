#include "mat.h"

std::default_random_engine Mat::engine;

void Mat::fromArray(const std::vector<Mat> &x, Mat &y)
{
    /* x: (N, 1, featureDim) */
    y = Mat(x.size(), x[0].cols);
    for (std::size_t i = 0; i < y.rows; i++) {
        for (std::size_t j = 0; j < y.cols; j++) {
            y(i, j) = x[i](1, j);
        }
    }
    return;
}

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
                y.val[i*cols + j] = 1;
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
    if (val.empty()) {
        val = std::vector<float>(r.val);
    } else {
        val.assign(r.val.begin(), r.val.end());
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
    val.swap(r.val);
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
                y.val[i*y.cols + j] += val[i*cols + k] * x.val[k*x.cols + j];
            }
        }
    }
    return y;
}

Mat Mat::operator /(const Mat &x) const
{
    Mat y(rows, cols);
    for (std::size_t i = 0; i < y.val.size(); i++) {
        y.val[i] = val[i]/x.val[i];
    }
    return y;
}

Mat Mat::operator +(const Mat &x) const
{
    Mat y(rows, cols);
    for (std::size_t i = 0; i < y.val.size(); i++) {
        y.val[i] = val[i] + x.val[i];
    }
    return y;
}

Mat Mat::operator -(const Mat &x) const
{
    Mat y(rows, cols);
    for (std::size_t i = 0; i < y.val.size(); i++) {
        y.val[i] = val[i] - x.val[i];
    }
    return y;
}

Mat Mat::operator %(const Mat &x) const
{
    Mat y(rows, cols);
    for (std::size_t i = 0; i < y.val.size(); i++) {
        y.val[i] = val[i] * x.val[i];
    }
    return y;
}

Mat &Mat::operator /=(const Mat &x)
{
    for (std::size_t i = 0; i < val.size(); i++) {
        val[i] /= x.val[i];
    }
    return *this;
}

Mat &Mat::operator +=(const Mat &x)
{
    for (std::size_t i = 0; i < val.size(); i++) {
        val[i] += x.val[i];
    }
    return *this;
}

Mat &Mat::operator -=(const Mat &x)
{
    for (std::size_t i = 0; i < val.size(); i++) {
        val[i] -= x.val[i];
    }
    return *this;
}

Mat &Mat::operator %=(const Mat &x)
{
    for (std::size_t i = 0; i < val.size(); i++) {
        val[i] *= x.val[i];
    }
    return *this;
}

Mat Mat::operator *(float x) const
{
    Mat y(rows, cols);
    for (std::size_t i = 0; i < val.size(); i++) {
        y.val[i] = val[i] * x;
    }
    return y;
}

Mat Mat::operator /(float x) const
{
    Mat y(rows, cols);
    for (std::size_t i = 0; i < val.size(); i++) {
        y.val[i] = val[i] / x;
    }
    return y;
}

Mat Mat::operator +(float x) const
{
    Mat y(rows, cols);
    for (std::size_t i = 0; i < val.size(); i++) {
        y.val[i] = val[i] + x;
    }
    return y;
}

Mat Mat::operator -(float x) const
{
    Mat y(rows, cols);
    for (std::size_t i = 0; i < val.size(); i++) {
        y.val[i] = val[i] - x;
    }
    return y;
}

Mat &Mat::operator *=(float x)
{
    for (std::size_t i = 0; i < val.size(); i++) {
        val[i] *= x;
    }
    return *this;
}

Mat &Mat::operator /=(float x)
{
    for (std::size_t i = 0; i < val.size(); i++) {
        val[i] /= x;
    }
    return *this;
}

Mat &Mat::operator +=(float x)
{
    for (std::size_t i = 0; i < val.size(); i++) {
        val[i] += x;
    }
    return *this;
}

Mat &Mat::operator -=(float x)
{
    for (std::size_t i = 0; i < val.size(); i++) {
        val[i] -= x;
    }
    return *this;
}

Mat Mat::tr() const
{
    Mat y(cols, rows);
    for (std::size_t i = 0; i < y.rows; i++) {
        for (std::size_t j = 0; j < y.cols; j++) {
            y.val[i*y.cols + j] = val[j*cols + i];
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
            y.val[i*y.cols + j] = val[(i + pr)*cols + j + pc];
        }
    }
    return y;
}

Mat Mat::flatten() const
{
    Mat y(rows*cols, 1, val);
    return y;
}

Mat Mat::f(std::function<float(float)> func) const
{
    Mat y(rows, cols);
    for (std::size_t i = 0; i < val.size(); i++) {
        y.val[i] = func(val[i]);
    }
    return y;
}

void Mat::set(std::size_t pr, std::size_t pc, const Mat &x)
{
    for (std::size_t i = pr; i < pr + x.rows; i++) {
        for (std::size_t j = pc; j < pc + x.cols; j++) {
            val[i*cols + j] = x.val[(i - pr)*cols + j - pc];
        }
    }
    return;
}

void Mat::setRow(size_t i, const std::vector<float> &row)
{
    for (std::size_t j = 0; j < cols; j++) {
        val[i*cols + j] = row[j];
    }
    return;
}

void Mat::setColumn(size_t j, const std::vector<float> &col)
{
    for (std::size_t i = 0; i < rows; i++) {
        val[i*cols + j] = col[i];
    }
    return;
}

void Mat::zero()
{
    val.assign(totalSize, 0);
    return;
}

void Mat::fill(float x)
{
    val.assign(totalSize, x);
    return;
}

void Mat::show() const
{
    std::cout<<"row:"<<rows<<", cols:"<<cols<<std::endl;
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            std::size_t index = i*cols + j;
            std::cout<<val[index]<<" ";
        }
        std::cout<<std::endl;
    }
    return;
}

size_t Mat::argmax() const
{
    float maxValue = val[0];
    std::size_t index = 0;
    for (std::size_t i = 1; i < totalSize; i++) {
        if (val[i] > maxValue) {
            maxValue = val[i];
            index = i;
        }
    }
    return index;
}

size_t Mat::argmin() const
{
    float minValue = val[0];
    std::size_t index = 0;
    for (std::size_t i = 1; i < totalSize; i++) {
        if (val[i] < minValue) {
            minValue = val[i];
            index = i;
        }
    }
    return index;
}

float Mat::max() const
{
    float maxValue = val[0];
    for (std::size_t i = 1; i < totalSize; i++) {
        if (val[i] > maxValue) {
            maxValue = val[i];
        }
    }
    return maxValue;
}

float Mat::min() const
{
    float minValue = val[0];
    for (std::size_t i = 1; i < totalSize; i++) {
        if (val[i] < minValue) {
            minValue = val[i];
        }
    }
    return minValue;
}

float Mat::sum() const
{
    float s = 0;
    for (std::size_t i = 0; i < totalSize; i++) {
        s += val[i];
    }
    return s;
}

float Mat::mean() const
{
    return sum()/float(totalSize);
}

float Mat::variance() const
{
    float u = mean();
    float s = 0;
    for (std::size_t i = 0; i < totalSize; i++) {
        s += (val[i] - u)*(val[i] - u);
    }
    return s;
}

void Mat::div(Mat &y, const Mat &x1, const Mat &x2)
{
    for (std::size_t i = 0; i < x1.totalSize; i++) {
        y.val[i] = x1.val[i] / x2.val[i];
    }
    return;
}

void Mat::add(Mat &y, const Mat &x1, const Mat &x2)
{
    for (std::size_t i = 0; i < x1.totalSize; i++) {
        y.val[i] = x1.val[i] + x2.val[i];
    }
    return;
}

void Mat::minus(Mat &y, const Mat &x1, const Mat &x2)
{
    for (std::size_t i = 0; i < x1.totalSize; i++) {
        y.val[i] = x1.val[i] - x2.val[i];
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
                    y(i*x2.cols + k, j*x2.cols + l) = x1.val[i*x1.cols + j] * x2.val[k*x2.cols + l];
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
                y.val[i*y.cols + j] += x1.val[i*x1.cols + k]*x2.val[k*x2.cols + j];
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
                y.val[i*y.cols + j] += x1.val[i*x1.cols + k]*x2.val[j*x2.cols + k];
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
                y.val[i*y.cols + j] += x1.val[k*x1.cols + i]*x2.val[k*x2.cols + j];
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
                y.val[i*y.cols + j] += x1.val[k*x1.cols + i] * x2.val[j*x2.cols + k];
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
        x.val[i] = std::atof(data.c_str());
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


