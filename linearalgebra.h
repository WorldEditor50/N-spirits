#ifndef LINEARALGEBRA_H
#define LINEARALGEBRA_H
#include "mat.h"
#include "func.h"

namespace LinearAlgebra {
void transpose(Mat &x);
/* trace */
int trace(const Mat& x, float &value);
/* diag */
void diag(const Mat &x, Mat &elements, int k);
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

struct SVD {
    static float eps;
    static std::size_t maxEpoch;
    static float normalize(Mat &x);
    static int solve(const Mat &x, Mat &u, Mat &s, Mat &v);
};

}
#endif // LINEARALGEBRA_H
