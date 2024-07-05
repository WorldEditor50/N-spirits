#include "linalg.h"

std::random_device LinAlg::Random::device;
std::default_random_engine LinAlg::Random::engine(LinAlg::Random::device());
std::mt19937 LinAlg::Random::generator(LinAlg::Random::device());

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
        y.val[i] = x1.val[i] - x2.val[i];
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

Tensor LinAlg::lerp(const Tensor &x1, const Tensor &x2, float alpha)
{
    Tensor x(x1.shape);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        x[i] = (1 - alpha)*x1[i] + alpha*x2[i];
    }
    return x;
}

void LinAlg::lerp(Tensor &x1, const Tensor &x2, float alpha)
{
    for (std::size_t i = 0; i < x1.totalSize; i++) {
        x1[i] = (1 - alpha)*x1[i] + alpha*x2[i];
    }
    return;
}

float LinAlg::Interplate::lagrange(const Tensor &x, const Tensor &y, float xi, int n)
{
    /*
        (xi, yi): interpolation point
        li(x) = ∏（x - xi)/(xi - xj), i!=j
        Ln(x) = Σ li(x)*yi
    */
    float ln = 0;
    for (int i = 0; i < n; i++) {
        float li = 1;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                li *= (xi - x[j])/(x[i] - x[j]);
            }
        }
        ln += li*y[i];
    }
    return ln;
}

float LinAlg::Interplate::newton(const Tensor &x, const Tensor &y, float xi, int n)
{
    float s = 0;
    int N = x.totalSize;
    Tensor delta(N, N);
    for (int i = 0; i < N; i++) {
        delta(0, i) = y[i];
    }
    /* first order difference */
    for (int i = 1; i < N; i++) {
        delta(1, i) = (delta(0, i) - delta(0, i - 1))/(x[i] - x[i - 1]);
    }
    /* high order difference */
    for (int i = 2; i < N; i++) {
        for (int j = i; j < N; j++) {
            delta(i, j) = (delta(i - 1, j) - delta(i - 1, j - 1))/(x[i] - x[0]);
        }
    }
    /* approximate */
    for (int i = 0; i < n; i++) {
        float p = 1;
        for (int j = 0; j < n; j++) {
            p *= xi - x[j];
        }
        s += p*delta(i, i);
    }
    return s;
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
        float d = x1.val[i] - x2.val[i];
        s += d*d;
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

float LinAlg::product(const Tensor &x)
{
    float p = 1;
    for (std::size_t i = 0; i < x.totalSize; i++) {
        p *= x[i];
    }
    return p;
}

float LinAlg::cosine(const Tensor &x1, const Tensor &x2)
{
    float s = 0;
    float s1 = 0;
    float s2 = 0;
    for (std::size_t i = 0; i < x1.totalSize; i++) {
        s += x1[i]*x2[i];
        s1 += x1[i]*x1[i];
        s2 += x2[i]*x2[i];
    }
    return s/std::sqrt(s1*s2);
}

float LinAlg::Kernel::rbf(const Tensor &x1, const Tensor &x2, float gamma)
{
    float s = 0;
    for (std::size_t i = 0; i < x1.totalSize; i++) {
        float d = x1.val[i] - x2.val[i];
        s += d*d;
    }
    return std::exp(-0.5*gamma*s);
}

float LinAlg::Kernel::laplace(const Tensor &x1, const Tensor &x2, float gamma)
{
    return std::exp(-0.5*gamma*normL2(x1, x2));
}

float LinAlg::Kernel::tanh(const Tensor &x1, const Tensor &x2, float c1, float c2)
{
    return std::tanh(c1*dot(x1, x2) + c2);
}

float LinAlg::Kernel::polynomial(const Tensor& x1, const Tensor& x2, float d, float p)
{
    return std::pow(LinAlg::dot(x1, x2) + d, p);
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

float LinAlg::gaussian(const Tensor &xi, const Tensor &ui, const Tensor &sigmai)
{

    return 0;
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
    /* x(n, 1) -> d(n, n) */
    int n = x.shape[0];
    Tensor d(n, n);
    for (int i = 0; i < n; i++) {
        d(i, i) = x[i];
    }
    return d;
}

Tensor LinAlg::diagInv(const Tensor &x)
{
    Tensor ix(x.shape);
    int n = x.shape[0];
    for (int i = 0; i < n; i++) {
        ix(i, i) = 1.0/x(i, i);
    }
    return ix;
}

int LinAlg::invert(const Tensor &x, Tensor &ix)
{
    /* gauss-jordan */
    if (x.shape[0] != x.shape[1]) {
        return -1;
    }
    int N = x.shape[0];
    ix = x;
    std::vector<int> row(N, 0);
    std::vector<int> col(N, 0);
    for (int k = 0; k < N; k++) {
        float d = 0;
        for (int i = k; i < N; i++) {
            for (int j = k; j < N; j++) {
                float p = std::abs(ix(i, j));
                if (p > d) {
                    d = p;
                    row[k] = i;
                    col[k] = j;
                }
            }
        }
        if (d + 1 == 1) {
            return -2;
        }
        if (row[k] != k) {
            exchangeCol(ix, k, col[k]);
        }
        if (col[k] != k) {
            exchangeRow(ix, k, row[k]);
        }
        ix(k, k) = 1.0/ix(k, k);

        for (int j = 0; j < N; j++) {
            if (j != k) {
                ix(k, j) *= ix(k, k);
            }
        }

        for (int i = 0; i < N; i++) {
            if (i != k) {
                for (int j = 0; j < N; j++) {
                    if (j != k) {
                        ix(i, j) -= ix(i, k)*ix(k, j);
                    }
                }
            }
        }

        for (int i = 0; i < N; i++) {
            if (i != k) {
                ix(i, k) *= -ix(k, k);
            }
        }

    }

    for (int k = N - 1; k >= 0; k--) {
        if (col[k] != k) {
            exchangeRow(ix, k, row[k]);
        }
        if (row[k] != k) {
            exchangeCol(ix, k, col[k]);
        }
    }
    return 0;
}

Tensor LinAlg::inv(const Tensor &x)
{
    Tensor ix(x.shape);
    invert(x, ix);
    return ix;
}

float LinAlg::eigen(const Tensor &x, Tensor &vec, int maxIterateCount, float eps)
{
    if (x.shape[0] != x.shape[1]) {
        return -1;
    }
    int N = x.shape[0];
    vec = Tensor(N, 1);
    vec.fill(1.0);
    float value = 0;
    float value_ = 0;
    Tensor vec_(N, 1);
    for (int it = 0; it < maxIterateCount; it++) {
        vec_.zero();
        Tensor::MM::ikkj(vec_, x, vec);
        vec = vec_;
        value_ = value;
        value = vec.max();
        vec /= value;
        float delta = std::sqrt((value - value_)*(value - value_));
        if (delta < eps) {
            break;
        }
    }
    return value;
}

int LinAlg::eigen(const Tensor &x, Tensor &vec, Tensor &value, int maxIterateCount, float eps)
{
    if (x.shape[0] != x.shape[1]) {
        return -1;
    }
    /* jacobi iteration */
    int N = x.shape[0];
    Tensor a = x;
    vec = eye(N);
    value = Tensor(N, 1);

    for (int it = 0; it < maxIterateCount; it++) {
        /* find max value on non pivot position */
        float maxVal = 0;
        int p = 0;
        int q = 0;
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N; j++) {
                float val = std::abs(a(i, j));
                if (i != j && val > maxVal) {
                    maxVal = val;
                    p = j;
                    q = i;
                }
            }
        }
        if (maxVal < eps) {
            //std::cout<<"iterate count:"<<it<<std::endl;
            break;
        }
        float app = a(p, p);
        float apq = a(p, q);
        float aqq = a(q, q);
        /* rotate */
        float theta = 0.5*std::atan2(-2*apq, aqq - app);
        float sinTheta = std::sin(theta);
        float cosTheta = std::cos(theta);
        float sin2Theta = std::sin(2*theta);
        float cos2Theta = std::cos(2*theta);

        a(p, p) = app*cosTheta*cosTheta + aqq*sinTheta*sinTheta + 2*apq*cosTheta*sinTheta;
        a(q, q) = app*sinTheta*sinTheta + aqq*cosTheta*cosTheta - 2*apq*cosTheta*sinTheta;
        a(p, q) = 0.5*(aqq- app)*sin2Theta + apq*cos2Theta;
        a(q, p) = a(p, q);

        for (int i = 0; i < N; i ++) {
            if (i != q && i != p) {
                float u = a(i, p);
                float v = a(i, q);
                a(i, p) = v*sinTheta + u*cosTheta;
                a(i, q) = v*cosTheta - u*sinTheta;
            }
        }

        for (int j = 0; j < N; j ++) {
            if (j != q && j != p) {
                float u = a(p, j);
                float v = a(q, j);
                a(p, j) = v*sinTheta + u*cosTheta;
                a(q, j) = v*cosTheta - u*sinTheta;
            }
        }

        /* eigen vector */
        for (int i = 0; i < N; i ++) {
            float u = vec(i, p);
            float v = vec(i, q);
            vec(i, p) = v*sinTheta + u*cosTheta;
            vec(i, q) = v*cosTheta - u*sinTheta;
        }

    }
    /* eigen value */
    for (int i = 0; i < N; i ++) {
        value[i] = a(i, i);
    }
    /* sign */
    for(int i = 0; i < N; i ++) {
        float s = 0;
        for(int j = 0; j < N; j ++) {
            s += vec(j, i);
        }
        if (s < 0) {
            for(int j = 0; j < N; j ++) {
                vec(j, i) *= -1;
            }
        }
    }
    return 0;
}

void LinAlg::xTAx(Tensor &y, const Tensor &x, const Tensor &a)
{
    /*
        x:(m, n)
        xT:(n, m)
        A:(m, m)
        xTAx:(n, n)
    */
    Tensor xTa(x.shape[1], x.shape[0]);
    Tensor::MM::kikj(xTa, x, a);
    Tensor::MM::ikkj(y, xTa, x);
    return;
}

void LinAlg::USVT(Tensor &y, const Tensor &u, const Tensor &s, const Tensor &v)
{
    Tensor r(u.shape[1], s.shape[0]);
    Tensor::MM::ikkj(r, u, s);
    Tensor::MM::ikjk(y, r, v);
    return;
}

Tensor LinAlg::USVT(const Tensor &u, const Tensor &s, const Tensor &v)
{
    Tensor r(u.shape[1], s.shape[0]);
    Tensor::MM::ikkj(r, u, s);
    Tensor y(r.shape[1], v.shape[0]);
    Tensor::MM::ikjk(y, r, v);
    return y;
}

void LinAlg::GaussianElimination::solve(const Tensor &a, Tensor &u)
{
    u = Tensor(a);
    std::size_t pivot = 0;
    std::size_t rows = u.shape[0];
    std::size_t cols = u.shape[1];
    for (std::size_t i = 0; i < rows && pivot < cols; i++, pivot++) {
        if (i >= cols) {
            /* found the pivot */
            break;
        }
        /* swap the row with pivot */
        for (std::size_t k = i + 1; k < rows; k++) {
            if (u(i, pivot) != 0) {
                break;
            } else if (k < rows) {
                exchangeRow(u, i, k);
            }
            if (k == rows - 1 && pivot < cols - 1 && u(i, pivot) == 0) {
                pivot++;
                k = i;
                continue;
            }
        }
        for (std::size_t k = i + 1; k < rows; k++) {
            double scalingFactor = u(k, pivot) / u(i, pivot);
            if (scalingFactor != 0) {
                u(k, pivot) = 0;
                for (std::size_t j = pivot + 1; j < cols; j++) {
                    u(k, j) -= scalingFactor * u(i, j);
                }
            }
        }
    }
    return;
}

void LinAlg::GaussianElimination::evaluate(const Tensor &u, Tensor &x)
{
    Tensor a(u);
    int N = a.shape[0];
    for (int i = N - 1; i >= 0; i--) {
        for (int j = i + 1; j < N; j++) {
            a(i, N) -= a(i, j) * a(j, N);
        }
        a(i, N) /= a(i, i);
    }
    for (int i = 0; i < N; i++) {
        x[i] = a(i, N);
    }
    return;
}

int LinAlg::gaussSeidel(const Tensor &a, const Tensor &b, Tensor &x, int iteration, float eps)
{
    int N = x.shape[0];
    /* pivot is not 0 */
    for (int i = 0; i < N; i++) {
        if (a(i, i) == 0) {
            return -1;
        }
    }
    /* positive definite matrix */
    Tensor l(a.shape);
    if (Cholesky::solve(a, l) == -2) {
        return -2;
    }
    Tensor x_(x.shape);
    for (int k = 0; k < iteration; k++) {
        float delta = 0.0;
        for (int i = 0; i < N; i++) {
            float s = 0;
            for (int j = 0; j < i; j++) {
                s += a(i, j)*x_[j];
            }
            for (int j = i + 1; j < N; j++) {
                s += a(i, j)*x[j];
            }
            x_[i] = (b[i] - s)/a(i, i);
            float d = (x_[i] - x[i]);
            delta += d*d;
        }
        delta = std::sqrt(delta/float(N));
        //std::cout<<"delta:"<<delta<<std::endl;
        if (delta < eps) {
            break;
        }
        x = x_;
    }
    return 0;
}

float LinAlg::det(const Tensor &x)
{
    float value = 0;
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
    return value;
}

int LinAlg::rank(const Tensor &x)
{
    Tensor xi(x.shape);
    GaussianElimination::solve(x, xi);
    //xi.printValue2D();
    int r = xi.shape[0];
    for (int i = 0; i < xi.shape[0]; i++) {
        int s = 0;
        for (int j = 0; j < xi.shape[1]; j++) {
            if (xi(i, j) == 0) {
                s++;
            }
        }
        if (s == xi.shape[1]) {
            r = i;
        }
    }
    return r;
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

int LinAlg::QR::iterate(const Tensor &x, Tensor &q, Tensor &r)
{
    int N = x.shape[0];
    q = LinAlg::eye(N);
    r = x;
    for (int k = 0; k < N - 1; k++) {
        int i = k + 1;
        float &rkk = r(k, k);
        float &rik = r(i, k);
        float tau = std::sqrt(rkk*rkk + rik*rik);
        float c = rkk/tau;
        float s = -rik/tau;
        for (int j = k; j < N; j++) {
            float &rij = r(i, j);
            float rkj = r(k, j);
            r(k, j) = c*rkj - s*rij;
            rij = s*rkj + c*rij;
        }
        r(i, k) = 0;

        for (int j = 0; j < N; j++) {
            float &qji = q(j, i);
            float qjk = q(j, k);
            q(j, k) = c*qjk - s*qji;
            qji = s*qjk + c*qji;
        }
    }
    return 0;
}

int LinAlg::QR::eigen(const Tensor &x, Tensor &e, float eps)
{
    Tensor p = LinAlg::eye(x.shape[0]);

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
    s = Tensor(x.shape);
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

int LinAlg::Cholesky::solve(const Tensor &x, Tensor &l)
{
    if (x.shape[0] != x.shape[1]) {
        return -1;
    }
    int n = x.shape[0];
    l = Tensor(n, n);
    for (int k = 0;  k < n; k++) {
        bool isReal = true;
        if (k == 0) {
            if (x(k, k) >= 0) {
                l(k, k) = std::sqrt(x(k, k));
            } else {
                isReal = false;
            }
        } else {
            float s = 0;
            for (int i = 0; i < k; i++) {
                s += l(k, i)*l(k, i);
            }
            float d = x(k, k) - s;
            if (d >= 0) {
                l(k, k) = std::sqrt(d);
            } else {
                isReal = false;
            }
        }
        /* positive definite matrix */
        if (!isReal || l(k, k) <= 0) {
            return -2;
        }

        if (k == 0) {
            for (int i = k + 1; i < n; i++) {
                l(i, k) = x(i, k)/l(k, k);
            }
        } else {
            for (int i = k + 1; i < n; i++) {
                float s = 0;
                for (int j = 0; j < k; j++) {
                    s += l(i, j)*l(k, j);
                }
                l(i, k) = (x(i, k) - s)/l(k, k);
            }
        }
    }
    return 0;
}

void LinAlg::PCA::solve(const Tensor &x, Tensor &u)
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

void LinAlg::PCA::project(const Tensor &x, const Tensor &u, int k, Tensor &y)
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
