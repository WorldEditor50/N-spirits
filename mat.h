#ifndef MAT_H
#define MAT_H
#include <vector>
#include <random>
#include <functional>
#include <cmath>
#include <ctime>
#include <string>
#include <sstream>
#include <iostream>

class Mat
{
public:
    enum Code {
        MAT_OK = 0
    };
    class Size
    {
    public:
        std::size_t rows;
        std::size_t cols;
    public:
        Size():rows(0),cols(0){}
        explicit Size(std::size_t r, std::size_t c)
            :rows(r),cols(c){}
    };

public:
    std::size_t rows;
    std::size_t cols;
    std::size_t totalSize;
    std::vector<float> val;
    static std::default_random_engine engine;
public:
    Mat(){}
    explicit Mat(std::size_t rows_, std::size_t cols_)
        :rows(rows_),cols(cols_),totalSize(rows_*cols_),val(rows_*cols_, 0){}
    explicit Mat(std::size_t rows_, std::size_t cols_, float value)
        :rows(rows_),cols(cols_),totalSize(rows_*cols_),val(rows_*cols_, value){}
    explicit Mat(const Size &s)
        :rows(s.rows),cols(s.cols),totalSize(s.rows*s.cols),val(s.rows*s.cols, 0){}
    explicit Mat(std::size_t rows_, std::size_t cols_, const std::vector<float> &data_)
        :rows(rows_),cols(cols_),totalSize(rows_*cols_),val(data_){}
    Mat(const Mat &r)
        :rows(r.rows),cols(r.cols),totalSize(r.totalSize),val(r.val){}
    Mat(Mat &&r)
        :rows(r.rows),cols(r.cols),totalSize(r.totalSize)
    {
        val.swap(r.val);
        r.rows = 0;
        r.cols = 0;
        r.totalSize = 0;
    }
    static Mat zeros(std::size_t rows, std::size_t cols);
    static Mat identity(std::size_t rows, std::size_t cols);
    inline Size size() const {return Size(rows, cols);}
    inline float &operator[](std::size_t i){return val[i];}
    inline float operator[](std::size_t i) const {return val[i];}
    inline float operator()(std::size_t i, std::size_t j) const {return val[i*cols + j];}
    inline float &operator()(std::size_t i, std::size_t j) {return val[i*cols + j];}
    Mat &operator=(const Mat &r);
    Mat &operator=(Mat &&r);
    /* object operation */
    Mat operator * (const Mat &x) const;
    Mat operator / (const Mat &x) const;
    Mat operator + (const Mat &x) const;
    Mat operator - (const Mat &x) const;
    Mat operator % (const Mat &x) const;
    Mat &operator /= (const Mat &x);
    Mat &operator += (const Mat &x);
    Mat &operator -= (const Mat &x);
    Mat &operator %= (const Mat &x);
    Mat operator * (float x) const;
    Mat operator / (float x) const;
    Mat operator + (float x) const;
    Mat operator - (float x) const;
    Mat &operator *= (float x);
    Mat &operator /= (float x);
    Mat &operator += (float x);
    Mat &operator -= (float x);
    Mat tr() const;
    int reshape(std::size_t rows_, std::size_t cols_);
    Mat sub(std::size_t sr, std::size_t sc, std::size_t offsetr, std::size_t offsetc) const;
    Mat flatten() const;
    Mat f(std::function<float(float)> func) const;
    void set(std::size_t sr, std::size_t sc, const Mat &x);
    void setRow(std::size_t i, const std::vector<float> &row);
    void setColumn(std::size_t j, const std::vector<float> &col);
    void zero();
    void fill(float x);
    void show() const;
    std::size_t argmax() const;
    std::size_t argmin() const;
    float max() const;
    float min() const;
    float sum() const;
    float mean() const;
    float variance() const;
    /* static operation */
    static void div(Mat &y, const Mat &x1, const Mat &x2);
    static void add(Mat &y, const Mat &x1, const Mat &x2);
    static void minus(Mat &y, const Mat &x1, const Mat &x2);

    static Mat kronecker(const Mat &x1, const Mat &x2);
    /* swap */
    struct Swap {
        static void row(Mat &x, std::size_t ri, std::size_t rj);
        static void col(Mat &x, std::size_t ci, std::size_t cj);
    };
    struct Multiply {
        static void ikkj(Mat &y, const Mat &x1, const Mat &x2);
        static void ikjk(Mat &y, const Mat &x1, const Mat &x2);
        static void kikj(Mat &y, const Mat &x1, const Mat &x2);
        static void kijk(Mat &y, const Mat &x1, const Mat &x2);
    };

    /* csv interface */
    static void parse(std::istringstream &stream, std::size_t cols, Mat &x);
    static void toString(const Mat &x, std::string &line);
};


#endif // MAT_H
