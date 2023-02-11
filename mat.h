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
    std::vector<float> data;
    static std::default_random_engine engine;
public:
    Mat(){}
    explicit Mat(std::size_t rows_, std::size_t cols_)
        :rows(rows_),cols(cols_),totalSize(rows_*cols_),data(rows_*cols_, 0){}
    explicit Mat(std::size_t rows_, std::size_t cols_, float value)
        :rows(rows_),cols(cols_),totalSize(rows_*cols_),data(rows_*cols_, value){}
    explicit Mat(const Size &s)
        :rows(s.rows),cols(s.cols),totalSize(s.rows*s.cols),data(s.rows*s.cols, 0){}
    explicit Mat(std::size_t rows_, std::size_t cols_, const std::vector<float> &data_)
        :rows(rows_),cols(cols_),totalSize(rows_*cols_),data(data_){}
    Mat(const Mat &r)
        :rows(r.rows),cols(r.cols),totalSize(r.totalSize),data(r.data){}
    Mat(Mat &&r)
        :rows(r.rows),cols(r.cols),totalSize(r.totalSize)
    {
        data.swap(r.data);
        r.rows = 0;
        r.cols = 0;
        r.totalSize = 0;
    }
    static Mat zeros(std::size_t rows, std::size_t cols);
    static Mat identity(std::size_t rows, std::size_t cols);
    inline Size size() const {return Size(rows, cols);}
    inline float &operator[](std::size_t i){return data[i];}
    inline float operator[](std::size_t i) const {return data[i];}
    inline float operator()(std::size_t i, std::size_t j) const {return data[i*cols + j];}
    inline float &operator()(std::size_t i, std::size_t j) {return data[i*cols + j];}
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
    void zero();
    void full(float x);
    void EMA(const Mat &r, float rho);

    void show() const;
    std::size_t argmax() const;
    std::size_t argmin() const;
    float max();
    float min();
    float sum();
    float mean();

    /* static operation */
    static float dot(const Mat &x1, const Mat &x2);
    static void div(Mat &y, const Mat &x1, const Mat &x2);
    static void add(Mat &y, const Mat &x1, const Mat &x2);
    static void minus(Mat &y, const Mat &x1, const Mat &x2);

    static float var(const Mat &x);
    static float var(const Mat &x, float u);
    static float cov(const Mat &x, const Mat &y);
    static float cov(const Mat &x, float ux, const Mat &y, float uy);
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
