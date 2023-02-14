#ifndef LINEARALGEBRA_H
#define LINEARALGEBRA_H
#include "mat.h"
#include "utils.h"

namespace LinearAlgebra {
void transpose(Mat &x);
/* trace */
int trace(const Mat& x, float &value);
/* diag */
void diag(const Mat &x, std::vector<float> &elements, int k);
/* gaussian elimination */
void gaussianElimination(const Mat &x, Mat &y);
/* det */
int det(const Mat &x, float &value);
/* rank */
std::size_t rank(const Mat &x);
/* LU */
namespace LU {
    int solve(const Mat &x, Mat& l, Mat &u);
    int inv(const Mat &x, Mat& xi);
};
/* QR */
namespace QR {
    int solve(const Mat &x, Mat &q, Mat &r);
    int eigen(const Mat &x, Mat &e);
};
/* svd */
namespace SVD {
    float normalize(Mat &x, float eps);
    float qrIteration(Mat &a, const Mat &q, float eps);
    int solve(const Mat &x, Mat &u, Mat &s, Mat &v, float eps=1e-7, std::size_t maxEpoch=1e6);
};
/* pca */
class PCA
{
public:
    Mat u;
public:
    PCA(){}
    void fit(const Mat &datas);
    void project(const Mat &x, size_t, Mat &y);
};
}
#endif // LINEARALGEBRA_H
