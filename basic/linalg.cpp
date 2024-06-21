#include "linalg.h"
std::default_random_engine LinAlg::Random::engine;

void LinAlg::add(Tensor &y, const Tensor &x1, const Tensor &x2)
{
    for (std::size_t i = 0; i < y.totalSize; i++) {
        y.val[i] = x1.val[i] + x2.val[i];
    }
    return;
}

void LinAlg::sub(Tensor &y, const Tensor &x1, const Tensor &x2)
{
    for (std::size_t i = 0; i < y.totalSize; i++) {
        y.val[i] = x1.val[i] + x2.val[i];
    }
    return;
}

void LinAlg::mul(Tensor &y, const Tensor &x1, const Tensor &x2)
{
    for (std::size_t i = 0; i < y.totalSize; i++) {
        y.val[i] = x1.val[i] * x2.val[i];
    }
    return;
}

void LinAlg::div(Tensor &y, const Tensor &x1, const Tensor &x2)
{
    for (std::size_t i = 0; i < y.totalSize; i++) {
        y.val[i] = x1.val[i] / x2.val[i];
    }
    return;
}

Tensor LinAlg::abs(const Tensor &x)
{
    Tensor y(x);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.val[i] = x.val[i] > 0 ? x.val[i] : -x.val[i];
    }
    return y;
}

Tensor LinAlg::sqrt(const Tensor &x)
{
    Tensor y(x);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.val[i] = std::sqrt(x.val[i]);
    }
    return y;
}

Tensor LinAlg::exp(const Tensor &x)
{
    Tensor y(x);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.val[i] = std::exp(x.val[i]);
    }
    return y;
}

Tensor LinAlg::sin(const Tensor &x)
{
    Tensor y(x);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.val[i] = std::sin(x.val[i]);
    }
    return y;
}

Tensor LinAlg::cos(const Tensor &x)
{
    Tensor y(x);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.val[i] = std::cos(x.val[i]);
    }
    return y;
}

Tensor LinAlg::tanh(const Tensor &x)
{
    Tensor y(x);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y.val[i] = std::tanh(x.val[i]);
    }
    return y;
}

void LinAlg::bernoulli(Tensor &x, float p)
{
    std::bernoulli_distribution distribution(p);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        x.val[i] = distribution(Random::engine);
    }
    return;
}

void LinAlg::gaussian(Tensor &x, float mu, float sigma)
{
    std::random_device device;
    std::default_random_engine engine(device());
    std::normal_distribution<float> distribution(mu, sigma);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        x.val[i] = distribution(engine);
    }
    return;
}

float LinAlg::normL1(const Tensor &x1, const Tensor &x2)
{
    float s = 0;
    for (std::size_t i = 0; i < x1.totalSize; i++) {
        s += x1.val[i] - x2.val[i];
    }
    return s;
}

float LinAlg::normL2(const Tensor &x1, const Tensor &x2)
{
    float s = 0;
    for (std::size_t i = 0; i < x1.totalSize; i++) {
        s += (x1.val[i] - x2.val[i])*(x1.val[i] - x2.val[i]);
    }
    return std::sqrt(s);
}

float LinAlg::normLp(const Tensor &x1, const Tensor &x2, float p)
{
    float s = 0;
    for (std::size_t i = 0; i < x1.totalSize; i++) {
        float delta = x1.val[i] - x2.val[i];
        s += std::pow(delta, p);
    }
    return std::pow(s, 1/p);
}

float LinAlg::normL8(const Tensor &x1, const Tensor &x2)
{
    float s = x1.val[0] - x2.val[0];
    for (std::size_t i = 1; i < x1.totalSize; i++) {
        float delta = x1.val[i] - x2.val[i];
        if (delta > s) {
            s = delta;
        }
    }
    return s;
}

float LinAlg::dot(const Tensor &x1, const Tensor &x2)
{
    float s = 0;
    for (std::size_t i = 0; i < x1.totalSize; i++) {
        s += x1[i]*x2[i];
    }
    return s;
}

void LinAlg::mean(const Tensor &x, Tensor &u)
{
    for (int i = 0; i < x.shape[1]; i++) {
        /* mean of column */
        for (int j = 0; j < x.shape[0]; j++) {
            u[i] += x(j, i);
        }
    }
    u /= float(x.shape[0]);
    return;
}

void LinAlg::variance(const Tensor &x, const Tensor &u, Tensor &sigma)
{
    for (int i = 0; i < x.shape[0]; i++) {
        for (int j = 0; j < x.shape[1]; j++) {
            float d = x(i, j) - u(0, j);
            sigma(0, j) += d*d;
        }
    }
    sigma /= float(x.shape[0]);
    return;
}

void LinAlg::mean(const std::vector<Tensor> &x, Tensor &u)
{
    for (std::size_t i = 0; i < x.size(); i++) {
        for (int j = 0; j < x[i].shape[0]; j++) {
            u[j] += x[i][j];
        }
    }
    u /= float(x.size());
    return;
}

void LinAlg::variance(const std::vector<Tensor> &x, const Tensor &u, Tensor &sigma)
{
    for (std::size_t i = 0; i < x.size(); i++) {
        for (int j = 0; j < x[i].shape[0]; j++) {
            float d = x[i][j] - u[j];
            sigma[j] += d*d;
        }
    }
    sigma /= float(x.size());
    return;
}

Tensor LinAlg::mean(const std::vector<Tensor> &x)
{
    Tensor u(1, x[0].shape[0]);
    for (std::size_t i = 0; i < x.size(); i++) {
        for (int j = 0; j < x[i].shape[0]; j++) {
            u[j] += x[i][j];
        }
    }
    u /= float(x.size());
    return u;
}

Tensor LinAlg::variance(const std::vector<Tensor> &x, const Tensor &u)
{
    Tensor sigma(u.shape);
    for (std::size_t i = 0; i < x.size(); i++) {
        for (int j = 0; j < x[i].shape[0]; j++) {
            float d = x[i][j] - u[j];
            sigma[j] += d*d;
        }
    }
    sigma /= float(x.size());
    return sigma;
}

void LinAlg::cov(const Tensor &x, Tensor &y)
{
    Tensor xi(x);
    for (int i = 0; i < x.shape[1]; i++) {
        /* mean of column */
        float u = 0;
        for (int j = 0; j < x.shape[0]; j++) {
            u += xi(j, i);
        }
        u /= float(x.shape[0]);
        /* delta: x - u */
        for (int j = 0; j < x.shape[0]; j++) {
            xi(j, i) -= u;
        }
    }
    /* y = A^T*A */
    Tensor::MM::kikj(y, xi, xi);
    y /= float(xi.shape[0]);
    return;
}

Tensor LinAlg::transpose(const Tensor &x)
{
    Tensor y(x.shape[1], x.shape[0]);
    for (int i = 0; i < x.shape[0]; i++) {
        for (int j = 0; j < x.shape[1]; j++) {
            y(j, i) = x(i, j);
        }
    }
    return y;
}

void LinAlg::exchangeRow(Tensor &x, int i1, int i2)
{
    for (int j = 0; j < x.shape[1]; j++) {
        float tmp = x(i1, j);
        x(i1, j) = x(i2, j);
        x(i2, j) = tmp;
    }
    return;
}

void LinAlg::exchangeCol(Tensor &x, int j1, int j2)
{
    for (int i = 0; i < x.shape[0]; i++) {
        float tmp = x(i, j1);
        x(i, j1) = x(i, j2);
        x(i, j2) = tmp;
    }
    return;
}

void LinAlg::embeddingRow(Tensor &x, int i, const Tensor &r)
{
    for (int j = 0; j < x.shape[1]; j++) {
        x(i, j) = r[j];
    }
    return;
}

void LinAlg::embeddingCol(Tensor &x, int j, const Tensor &c)
{
    for (int i = 0; i < x.shape[0]; i++) {
        x(i, j) = c[i];
    }
    return;
}

Tensor LinAlg::eye(int n)
{
    Tensor I(n, n);
    for (int i = 0; i < n; i++) {
        I(i, i) = 1;
    }
    return I;
}

int LinAlg::trace(const Tensor &x, float &value)
{
    if (x.shape[0] != x.shape[1]) {
        return -1;
    }
    std::size_t N = x.shape[1];
    value = 0;
    for (std::size_t i = 0; i < N; i++) {
        value += x(i, i);
    }
    return 0;
}

Tensor LinAlg::diag(const Tensor &x)
{
    std::size_t N = std::min(x.shape[0], x.shape[1]);
    Tensor d(N, 1);
    for (std::size_t i = 0; i < N; i++) {
        d[i] = x(i, i);
    }
    return d;
}

void LinAlg::gaussianElimination(const Tensor &x, Tensor &y)
{
    y = Tensor(x);
    std::size_t pivot = 0;
    std::size_t rows = y.shape[0];
    std::size_t cols = y.shape[1];
    for (std::size_t i = 0; i < rows && pivot < cols; i++, pivot++) {
        if (i >= cols) {
            /* found the pivot */
            break;
        }
        /* swap the row with pivot */
        for (std::size_t k = i + 1; k < rows; k++) {
            if (y(i, pivot) != 0) {
                break;
            } else if (k < rows) {
                exchangeRow(y, i, k);
            }
            if (k == rows - 1 && pivot < cols - 1 && y(i, pivot) == 0) {
                pivot++;
                k = i;
                continue;
            }
        }
        for (std::size_t k = i + 1; k < rows; k++) {
            double scalingFactor = y(k, pivot) / y(i, pivot);
            if (scalingFactor != 0) {
                y(k, pivot) = 0;
                for (std::size_t j = pivot + 1; j < cols; j++) {
                    y(k, j) -= scalingFactor * y(i, j);
                }
            }
        }
    }
    return;
}

int LinAlg::det(const Tensor &x, float &value)
{
    if (x.shape[0]!= x.shape[1]) {
        return -1;
    }
    /* 1-order */
    if (x.shape[0] == 1) {
        value = x(0, 0);
    }
    /* 2-order */
    if (x.shape[0] == 2) {
        value = x(0, 0)*x(1, 1) - x(0, 1)*x(1, 0);
    }
    /*
       n-order:
        Gaussian Elimination -> up triangle matrix -> det(x) = ∏ Diag(U)
    */
    Tensor d(x);
    std::size_t N = x.shape[1];
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
        /* exchange row */
        exchangeRow(d, i, t);
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

size_t LinAlg::rank(const Tensor &x)
{
    return 0;
}

int LinAlg::QR::solve(const Tensor &x, Tensor &q, Tensor &r)
{
    if (x.shape[0] != x.shape[1]) {
        return -1;
    }
    q = Tensor(x.shape);
    r = Tensor(x.shape);
    Tensor a(x);
    std::size_t N = x.shape[0];
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

int LinAlg::QR::eigen(const Tensor &x, Tensor &e)
{
    return 0;
}

int LinAlg::LU::solve(const Tensor &x, Tensor &l, Tensor &u)
{
    /* square matrix */
    if (x.shape[0] != x.shape[1]) {
        return -1;
    }
    /* U1i=X1i,Li1=Xi1/U11 */
    for (int i = 0; i < x.shape[0]; i++) {
         u(0, i) = x(0, i);
         l(i, 0) = x(i, 0)/u(0, 0);
    }

    int N = x.shape[0] - 1;
    for (int r = 1; r < x.shape[0]; r++) {
        for (int i = r; i < x.shape[1]; i++) {
            /* Σ(k=1->r-1)Lrk*Uki */
            float s = 0.0;
            for (int k = 0; k < r; k++) {
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
                for (int k = 0; k < r; k++) {
                    s += l(i, k) * u(k, r);
                }
                /* Lir = Xir - Σ(k=1->r-1)Lik*Ukr (i=r+1,r+2,...,n and r≠n) */
                l(i, r) = (x(i, r) - s)/u(r, r);
            }
        }
    }
    return 0;
}

int LinAlg::LU::inv(const Tensor &x, Tensor &xi)
{
    Tensor l(x.shape);
    Tensor u(x.shape);
    /* LU decomposition */
    int ret = LU::solve(x, l, u);
    if (ret < 0) {
        return -1;
    }
    /* check invertable */
    float prod = 1;
    for (int i = 1; i < x.shape[0]; i++) {
        prod *= u(i, i);
    }
    if (prod == 0) {
        return -2;
    }
    /* inverse matrix of u */
    Tensor ui(x.shape);
    for (int i = 0; i < ui.shape[0]; i++) {
        ui(i, i) = 1/u(i, i);
        for (int k = i - 1; k >= 0; k--) {
            float s = 0;
            for (int j = k + 1; j <= i; j++) {
                s += u(k, j)*ui(j, i);
            }
            ui(k, i) = -s/u(k, k);
        }
    }
    /* inverse matrix of l */
    Tensor li(x.shape);
    for (int i = 0; i < li.shape[1]; i++) {
        li(i, i) = 1;
        for (int k = i + 1; k < x.shape[0]; k++) {
            for (int j = i; j <= k - 1; j++) {
                li(k, i) -= l(k, j)*li(j, i);
            }
        }
    }
    /* inverse matrix of x, x = l*u , xi = ui*li */
    xi = Tensor(x.shape);
    Tensor::MM::ikkj(xi, ui, li);
    return 0;
}

float LinAlg::SVD::normalize(Tensor &x, float eps)
{
    float s = std::sqrt(dot(x, x));
    if (s < eps) {
        return 0;
    }
    x /= s;
    return s;
}

float LinAlg::SVD::qrIteration(Tensor &a, const Tensor &q, float eps)
{
    float r = normalize(a, eps);
    if (r < eps) {
        return r;
    }
    for (int i = 0; i < q.shape[0]; i++) {
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

int LinAlg::SVD::solve(const Tensor &x, Tensor &u, Tensor &s, Tensor &v, float eps, std::size_t maxEpoch)
{
    u = Tensor(x.shape[0], x.shape[0]);
    v = Tensor(x.shape[1], x.shape[1]);
    s = Tensor(x.size());
    /* column vectors */
    Tensor ur(x.shape[0], 1);
    Tensor nextUr(x.shape[0], 1);
    Tensor vr(x.shape[1], 1);
    Tensor nextVr(x.shape[1], 1);
    while (1) {
        uniform(ur, -1, 1);
        float s = normalize(ur,eps);
        if (s > eps) {
            break;
        }
    }
    std::size_t N = std::min(x.shape[0], x.shape[1]);
    for (std::size_t n = 0; n < N; n++){
        float r = -1;
        for (std::size_t epoch = 0; epoch < maxEpoch; epoch++) {
            nextUr.zero();
            nextVr.zero();
            /* nextVr = a^T * ur */
            Tensor::MM::kikj(nextVr, x, ur);
            /* QR: v */
            r = qrIteration(nextVr, v, eps);
            if (r < eps) {
                break;
            }
            /* nextUr = a * nextVr */
            Tensor::MM::ikkj(nextUr, x, nextVr);
            /* QR: u */
            r = qrIteration(nextUr, u, eps);
            if (r < eps) {
                break;
            }
            /* error */
            float delta = normL2(nextUr, ur);
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
        embeddingRow(u, n, ur);
        embeddingRow(v, n, vr);
    }
    u = u.tr();
    v = v.tr();
    return 0;
}

void LinAlg::PCA::fit(const Tensor &x)
{
    /* covariance matrix */
    Tensor y(x.shape[1], x.shape[1]);
    cov(x, y);
    /* svd */
    Tensor s;
    Tensor v;
    SVD::solve(y, u, s, v);
    return;
}

void LinAlg::PCA::project(const Tensor &x, int k, Tensor &y)
{
    if (k >= u.shape[1]) {
        return;
    }

    /* reduce dimention */
    Tensor uh(u.shape[0], k);
    for (int i = 0; i < u.shape[0]; i++) {
        for (int j = 0; j < k; j++) {
            uh(i, j) = u(i, j);
        }
    }
    /* y = u^T * x */
    y = Tensor(x.shape[0], k);
    /* (1, 2) = (1, 13)*(13, 2)^T */
    Tensor::MM::ikkj(y, x, uh);
    return;
}
