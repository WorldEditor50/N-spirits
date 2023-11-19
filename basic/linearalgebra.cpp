#include "linearalgebra.h"

void LinearAlgebra::transpose(Mat &x)
{
    Mat y(x.cols, x.rows);
    for (std::size_t i = 0; i < x.rows; i++) {
        for (std::size_t j = 0; j < x.cols; j++) {
            y(j, i) = x(i, j);
        }
    }
    x.rows = y.rows;
    x.cols = y.cols;
    x.val.swap(y.val);
    return;
}

int LinearAlgebra::trace(const Mat &x, float &value)
{
    if (x.rows != x.cols) {
        return -1;
    }
    std::size_t N = x.cols;
    value = 0;
    for (std::size_t i = 0; i < N; i++) {
        value += x(i, i);
    }
    return 0;
}

void LinearAlgebra::diag(const Mat &x, std::vector<float> &elements, int k)
{
    std::size_t N = std::min(x.rows, x.cols);
    for (std::size_t i = 0; i < N; i++) {
         elements.push_back(x(i, i));
    }
    return;
}

void LinearAlgebra::gaussianElimination(const Mat &x, Mat &y)
{
    y = Mat(x);
    std::size_t pivot = 0;
    for (std::size_t i = 0; i < y.rows && pivot < y.cols; i++, pivot++) {
        if (i >= y.cols) {
            /* found the pivot */
            break;
        }
        /* swap the row with pivot */
        for (std::size_t k = i + 1; k < y.rows; k++) {
            if (y(i, pivot) != 0) {
                break;
            } else if (k < y.rows) {
                Mat::Swap::row(y, i, k);
            }
            if (k == y.rows - 1 && pivot < y.cols - 1 && y(i, pivot) == 0) {
                pivot++;
                k = i;
                continue;
            }
        }
        for (std::size_t k = i + 1; k < y.rows; k++) {
            double scalingFactor = y(k, pivot) / y(i, pivot);
            if (scalingFactor != 0) {
                y(k, pivot) = 0;
                for (std::size_t j = pivot + 1; j < y.cols; j++) {
                    y(k, j) -= scalingFactor * y(i, j);
                }
            }
        }
    }
    return;
}

int LinearAlgebra::det(const Mat &x, float &value)
{
    if (x.rows != x.cols) {
        return -1;
    }
    /* 1-order */
    if (x.rows == 1) {
        value = x(0, 0);
    }
    /* 2-order */
    if (x.rows == 2) {
        value = x(0, 0)*x(1, 1) - x(0, 1)*x(1, 0);
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
        /* swap row */
        Mat::Swap::row(d, i, t);
        if (t != i) {
            value *= -1;
        }
        /* ∏ Diag(U)  */
        float pivot = d(i, i);
        value *= pivot;
        for (std::size_t j = 0; j < N; j++) {
            d(i, j) /= pivot;
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

size_t LinearAlgebra::rank(const Mat &x)
{
    return 0;
}

int LinearAlgebra::QR::solve(const Mat &x, Mat &q, Mat &r)
{
    if (x.rows != x.cols) {
        return -1;
    }
    q = Mat(x.rows, x.cols);
    r = Mat(x.rows, x.cols);
    Mat a(x);
    std::size_t N = x.rows;
    for (std::size_t k = 0; k < N; k++) {
        /* normalize: Rkk = ||Aik|| */
        float s = 0;
        for (std::size_t i = 0; i < N; i++) {
            s += a(i, k) * a(i, k);
        }
        r(k, k) = std::sqrt(s);
        /* Qik = Aik / Rkk */
        for (std::size_t i = 0; i < N; i++) {
            q(i, k) = a(i, k) / r(k, k);
        }

        for (std::size_t i = k + 1; i < N; i++) {
            /* Rki = Aji * Qjk */
            for (std::size_t j = 0; j < N; j++) {
                r(k, i) += a(j, i) * q(j, k);
            }
            /* Aji = Rki * Qjk */
            for (std::size_t j = 0; j < N; j++) {
                a(j, i) -= r(k, i) * q(j, k);
            }
        }
    }
    return 0;
}

int LinearAlgebra::QR::eigen(const Mat &x, Mat &e)
{
    return 0;
}

int LinearAlgebra::LU::solve(const Mat &x, Mat &l, Mat &u)
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

    std::size_t N = x.rows - 1;
    for (std::size_t r = 1; r < x.rows; r++) {
        for (std::size_t i = r; i < x.cols; i++) {
            /* Σ(k=1->r-1)Lrk*Uki */
            float s = 0.0;
            for (std::size_t k = 0; k < r; k++) {
                s += l(r, k) * u(k, i);
            }
            /* Uri= Xri- Σ(k=1->r-1)Lrk*Uki (i=r,r+1,...,n) */
            u(r, i) = x(r, i) - s;
            if (i == r) {
                l(r, r) = 1;
            } else if (r == N) {
                l(N, N) = 1;
            } else {
                /* Σ(k=1->r-1)Lik*Ukr */
                float s = 0.0;
                for (std::size_t k = 0; k < r; k++) {
                    s += l(i, k) * u(k, r);
                }
                /* Lir = Xir - Σ(k=1->r-1)Lik*Ukr (i=r+1,r+2,...,n and r≠n) */
                l(i, r) = (x(i, r) - s)/u(r, r);
            }
        }
    }
    return 0;
}

int LinearAlgebra::LU::inv(const Mat &x, Mat &xi)
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
    Mat::Multi::ikkj(xi, ui, li);
    return 0;
}

float LinearAlgebra::SVD::normalize(Mat &x, float eps)
{
    float s = std::sqrt(util::dot(x, x));
    if (s < eps) {
        return 0;
    }
    x /= s;
    return s;
}

float LinearAlgebra::SVD::qrIteration(Mat &a, const Mat &q, float eps)
{
    float r = normalize(a, eps);
    if (r < eps) {
        return r;
    }
    for (std::size_t i = 0; i < q.rows; i++) {
        float s = 0;
        for (std::size_t j = 0; j < a.totalSize; j++) {
            s += q(i, j)*a[j];
        }
        for (std::size_t j = 0; j < a.totalSize; j++) {
            a[j] -= s*q(i, j);
        }
    }
    normalize(a, eps);
    return r;
}

int LinearAlgebra::SVD::solve(const Mat &x, Mat &u, Mat &s, Mat &v, float eps, std::size_t maxEpoch)
{
    u = Mat(x.rows, x.rows);
    v = Mat(x.cols, x.cols);
    s = Mat(x.size());
    /* column vectors */
    Mat ur(x.rows, 1);
    Mat nextUr(x.rows, 1);
    Mat vr(x.cols, 1);
    Mat nextVr(x.cols, 1);
    while (1) {
        util::uniform(ur);
        float s = normalize(ur,eps);
        if (s > eps) {
            break;
        }
    }
    std::size_t N = std::min(x.rows, x.cols);
    for (std::size_t n = 0; n < N; n++){
        float r = -1;
        for (std::size_t epoch = 0; epoch < maxEpoch; epoch++) {
            nextUr.zero();
            nextVr.zero();
            /* nextVr = a^T * ur */
            Mat::Multi::kikj(nextVr, x, ur);
            /* QR: v */
            r = qrIteration(nextVr, v, eps);
            if (r < eps) {
                break;
            }
            /* nextUr = a * nextVr */
            Mat::Multi::ikkj(nextUr, x, nextVr);
            /* QR: u */
            r = qrIteration(nextUr, u, eps);
            if (r < eps) {
                break;
            }
            /* error */
            float delta = util::Norm::l2(nextUr, ur);
            if (delta < eps) {
                break;
            }
            ur = nextUr;
            vr = nextVr;
        }
        if (r < eps) {
            break;
        }
        /* update */
        s(n, n) = r;
        u.setRow(n, ur.val);
        v.setRow(n, vr.val);
    }
    u = u.tr();
    v = v.tr();
    return 0;
}

void LinearAlgebra::PCA::fit(const Mat &datas)
{
    /* covariance matrix */
    Mat y(datas.cols, datas.cols);
    util::cov(y, datas);
    /* svd */
    Mat s;
    Mat v;
    SVD::solve(y, u, s, v);
    return;
}

void LinearAlgebra::PCA::project(const Mat &x, std::size_t k, Mat &y)
{
    if (k >= u.cols) {
        return;
    }

    /* reduce dimention */
    Mat uh(u.rows, k);
    for (std::size_t i = 0; i < u.rows; i++) {
        for (std::size_t j = 0; j < k; j++) {
            uh(i, j) = u(i, j);
        }
    }
    /* y = u^T * x */
    y = Mat(x.rows, k);
    /* (1, 2) = (1, 13)*(13, 2)^T */
    Mat::Multi::ikkj(y, x, uh);
    return;
}
