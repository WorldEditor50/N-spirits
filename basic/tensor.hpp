#ifndef TENSOR_H
#define TENSOR_H
#include <vector>
#include <functional>
#include <iostream>
#include <assert.h>

template<typename T>
class Tensor_
{
public:
    using ValueType = T;
    using Shape = std::vector<int>;
public:
    std::vector<T> val;
    std::vector<int> sizes;
    std::vector<int> shape;
    std::size_t totalSize;
 public:
    /* default construct */
    Tensor_():totalSize(0){}
    static void initParams(const std::vector<int> &shape, std::vector<int> &sizes, std::size_t &totalsize)
    {
        totalsize = 1;
        for (std::size_t i = 0; i < shape.size(); i++) {
            totalsize *= shape[i];
        }
        sizes = std::vector<int>(shape.size(), 1);
        for (std::size_t i = 0; i < shape.size() - 1; i++) {
            for (std::size_t j = i + 1; j < shape.size(); j++) {
                 sizes[i] *= shape[j];
            }
        }
        return;
    }
    /* contruct with shape */
    explicit Tensor_(const std::vector<int> &shape_):totalSize(1),shape(shape_)
    {
        initParams(shape, sizes, totalSize);
        val = std::vector<T>(totalSize, 0);
    }

    explicit Tensor_(const std::vector<int> &shape_, const std::initializer_list<T> &val_):
        totalSize(1),shape(shape_),val(val_)
    {
        initParams(shape, sizes, totalSize);
    }


    explicit Tensor_(const std::initializer_list<int> &shape_, const std::initializer_list<T> &val_):
        totalSize(1),shape(shape_),val(val_)
    {
        initParams(shape, sizes, totalSize);
    }

    /* construct with shape */
    template<typename ...Dim>
    explicit Tensor_(Dim ...dim):totalSize(1),shape({dim...})
    {
        initParams(shape, sizes, totalSize);
        val = std::vector<T>(totalSize, 0);
    }

    /* copy construct */
    Tensor_(const Tensor_ &r)
        :totalSize(r.totalSize),shape(r.shape),sizes(r.sizes),val(r.val){}

    /* assign operator */
    Tensor_ &operator=(const Tensor_ &r)
    {
        if (this == &r) {
            return *this;
        }
        totalSize = r.totalSize;
        shape = r.shape;
        sizes = r.sizes;
        val = r.val;
        return *this;
    }

    static Tensor_ zeros(Shape &shape)
    {
        Tensor_ x(shape);
        return x;
    }

    template<typename ...Dim>
    static Tensor_ zeros(Dim ...dim)
    {
        Tensor_ x(dim...);
        return x;
    }

    static Tensor_ ones(Shape &shape)
    {
        Tensor_ x(shape);
        x.fill(1);
        return x;
    }

    template<typename ...Dim>
    static Tensor_ ones(Dim ...dim)
    {
        Tensor_ x(dim...);
        x.fill(1);
        return x;
    }

    template<typename ...Dim>
    Tensor_ sub(Dim ...di)
    {
        Tensor_ y;
        std::vector<int> dims = std::vector<int>{di...};
        std::vector<int> subDims(shape.begin() + dims.size(), shape.end());
        y = Tensor_(subDims);
        std::size_t pos = 0;
        for (std::size_t i = 0; i < dims.size(); i++) {
            pos += sizes[i]*dims[i];
        }
        for (std::size_t i = 0; i < y.totalSize; i++) {
            y.val[i] = val[i + pos];
        }
        return y;
    }
    inline bool empty() const {return totalSize == 0;}
    void zero(){val.assign(totalSize, 0);}
    void fill(T value){val.assign(totalSize, value);}
    inline T &operator[](std::size_t i) {return val[i];}
    inline T operator[](std::size_t i) const {return val[i];}
    bool isShapeEqual(const Tensor_ &x) const
    {
        if (shape.size() != x.shape.size()) {
            return false;
        }
        for (int i = 0; i < shape.size(); i++) {
            if (shape[i] != x.shape[i]) {
                return false;
            }
        }
        return true;
    }

    /* visit */
    template<typename ...Index>
    inline int posOf(Index ...index) const
    {
        int indexs[] = {index...};
        int pos = 0;
        for (std::size_t i = 0; i < sizes.size(); i++) {
            pos += sizes[i]*indexs[i];
        }
        return pos;
    }

    inline int posOf(const std::vector<int> &indexs) const
    {
        int pos = 0;
        for (std::size_t i = 0; i < sizes.size(); i++) {
            pos += sizes[i]*indexs[i];
        }
        return pos;
    }

    inline void indexOf(int pos, std::vector<int> &indexs) const
    {
        /*
            shape: (2, 3, 4, 5)
            sizes: (60, 20, 5, 1)
            totalsize 2*3*4*5 = 120
            indexs:(1, 2, 3, 4)
            pos : 60*1 + 20*2 + 5*3 + 4*1 = 119

            i0 = pos/60
            i1 = (pos - i0*60)/20
            i2 = (pos - i0*60 - i1*20)/5
            i3 = pos - i0*60 - i1*20 - i2*5
        */
        int pos_ = 0;
        for (std::size_t i = 0; i < sizes.size(); i++) {
            indexs[i] = (pos - pos_)/sizes[i];
            pos_ += indexs[i]*sizes[i];
        }
        return;
    }

    inline std::vector<int> indexOf(int pos) const
    {
        std::vector<int> indexes(shape.size(), 0);
        indexOf(pos, indexes);
        return indexes;
    }

    template<typename ...Index>
    inline T &operator()(Index ...index) { return val[posOf(index...)]; }

    template<typename ...Index>
    inline T operator()(Index ...index) const { return val[posOf(index...)]; }

    inline T &operator()(const std::vector<int> &indexs) { return val[posOf(indexs)]; }

    inline T operator()(const std::vector<int> &indexs) const { return val[posOf(indexs)]; }

    template<typename ...Index>
    inline T &at(Index ...index) { return val[posOf(index...)]; }

    template<typename ...Dim>
    Tensor_ reshape(Dim ...dim) const
    {
        Tensor_ x(dim...);
        if (x.totalSize != totalSize) {
            return x;
        }
        x.val = val;
        return x;
    }

    template<typename ...Dim>
    static void reshape(Tensor_ &x, Dim ...dim)
    {
        std::vector<int> newShape = {dim...};
        /* size */
        int s = 1;
        for (std::size_t i = 0; i < x.shape.size(); i++) {
            s *= newShape[i];
        }
        if (s != x.totalSize) {
            return;
        }
        initParams(newShape, x.sizes, x.totalSize);
        x.shape = newShape;
        return;
    }

    Tensor_ flatten() const
    {
        Tensor_ x(totalSize);
        x.val = val;
        return x;
    }

    static void permuteIndexs(const std::vector<int> &indexs,
                              const std::vector<int> &permuteMap,
                              std::vector<int> &newIndexs)
    {
        for (std::size_t i = 0; i < permuteMap.size(); i++) {
            int k = permuteMap[i];
            newIndexs[i] = indexs[k];
        }
        return;
    }

    template<typename ...Pos>
    Tensor_ permute(Pos ...p) const
    {
        /*
            shape: [3, 2, 1]
            permute: (2, 1, 0)
            new shape: [1, 2, 3]
        */
        std::vector<int> permuteMap = {p...};
        /* permute shape */
        std::vector<int> newShape(shape.size(), 0);
        permuteIndexs(shape, permuteMap, newShape);
        Tensor_ x(newShape);
        /* permute value */
        std::vector<int> indexs(shape.size(), 0);
        std::vector<int> newIndexs(shape.size(), 0);
        for (std::size_t i = 0; i < val.size(); i++) {
            indexOf(i, indexs);
            permuteIndexs(indexs, permuteMap, newIndexs);
            x(newIndexs) = val[i];
        }
        return x;
    }

    Tensor_ tr() const
    {
        int r = shape[0];
        int c = shape[1];
        Tensor_ y(c, r);
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                y.val[j*c + i] = val[i*c + j];
            }
        }
        return y;
    }

    /* operator */
    Tensor_ operator +(const Tensor_ &x) const
    {
        Tensor_ y(shape);
        for (std::size_t i = 0; i < val.size(); i++) {
            y.val[i] = val[i] + x.val[i];
        }
        return y;
    }

    Tensor_ operator -(const Tensor_ &x) const
    {
        Tensor_ y(shape);
        for (std::size_t i = 0; i < val.size(); i++) {
            y.val[i] = val[i] - x.val[i];
        }
        return y;
    }

    Tensor_ operator *(const Tensor_ &x) const
    {
        Tensor_ y(shape);
        for (std::size_t i = 0; i < val.size(); i++) {
            y.val[i] = val[i] * x.val[i];
        }
        return y;
    }

    Tensor_ operator /(const Tensor_ &x) const
    {
        Tensor_ y(shape);
        for (std::size_t i = 0; i < val.size(); i++) {
            y.val[i] = val[i] / x.val[i];
        }
        return y;
    }

    Tensor_ &operator +=(const Tensor_ &x)
    {
        for (std::size_t i = 0; i < val.size(); i++) {
            val[i] += x.val[i];
        }
        return *this;
    }

    Tensor_ &operator -=(const Tensor_ &x)
    {
        for (std::size_t i = 0; i < val.size(); i++) {
            val[i] -= x.val[i];
        }
        return *this;
    }

    Tensor_ &operator *=(const Tensor_ &x)
    {
        for (std::size_t i = 0; i < val.size(); i++) {
            val[i] *= x.val[i];
        }
        return *this;
    }

    Tensor_ operator /=(const Tensor_ &x)
    {
        for (std::size_t i = 0; i < val.size(); i++) {
            val[i] /= x.val[i];
        }
        return *this;
    }

    Tensor_ operator +(T x) const
    {
        Tensor_ y(shape);
        for (std::size_t i = 0; i < val.size(); i++) {
            y.val[i] = val[i] + x;
        }
        return y;
    }

    Tensor_ operator -(T x) const
    {
        Tensor_ y(shape);
        for (std::size_t i = 0; i < val.size(); i++) {
            y.val[i] = val[i] - x;
        }
        return y;
    }

    Tensor_ operator *(T x) const
    {
        Tensor_ y(shape);
        for (std::size_t i = 0; i < val.size(); i++) {
            y.val[i] = val[i] * x;
        }
        return y;
    }

    Tensor_ operator /(T x) const
    {
        Tensor_ y(shape);
        for (std::size_t i = 0; i < val.size(); i++) {
            y.val[i] = val[i] / x;
        }
        return y;
    }

    Tensor_ &operator +=(T x)
    {
        for (std::size_t i = 0; i < val.size(); i++) {
            val[i] += x;
        }
        return *this;
    }

    Tensor_ &operator -=(T x)
    {
        for (std::size_t i = 0; i < val.size(); i++) {
            val[i] -= x;
        }
        return *this;
    }

    Tensor_ &operator *=(T x)
    {
        for (std::size_t i = 0; i < val.size(); i++) {
            val[i] *= x;
        }
        return *this;
    }

    Tensor_ &operator /=(T x)
    {
        for (std::size_t i = 0; i < val.size(); i++) {
            val[i] /= x;
        }
        return *this;
    }

    /* statistics */
    T sum() const
    {
        T s = 0;
        for (std::size_t i = 0; i < totalSize; i++) {
            s += val[i];
        }
        return s;
    }

    T mean() const
    {
        T s = sum();
        return s/T(totalSize);
    }

    T variance(T u) const
    {
        T s = 0;
        for (std::size_t i = 0; i < val.size(); i++) {
            s += (val[i] - u)*(val[i] - u);
        }
        return s/T(totalSize);
    }

    T max() const
    {
        T value = val[0];
        for (std::size_t i = 0; i < val.size(); i++) {
            if (value < val[i]) {
                value = val[i];
            }
        }
        return value;
    }

    T min() const
    {
        T value = val[0];
        for (std::size_t i = 0; i < val.size(); i++) {
            if (value > val[i]) {
                value = val[i];
            }
        }
        return value;
    }

    int argmax() const
    {
        T value = val[0];
        int index = 0;
        for (std::size_t i = 0; i < val.size(); i++) {
            if (value < val[i]) {
                value = val[i];
                index = i;
            }
        }
        return index;
    }

    int argmin() const
    {
        T value = val[0];
        int index = 0;
        for (std::size_t i = 0; i < val.size(); i++) {
            if (value > val[i]) {
                value = val[i];
                index = i;
            }
        }
        return index;
    }

    /* initialize */
    template<typename ...Value>
    void assign(Value ...value)
    {
        val = std::vector<T>{value...};
        return;
    }

    void normalize()
    {
        double minValue = val[0];
        double maxValue = val[0];
        for (std::size_t i = 0; i < val.size(); i++) {
            if (minValue > val[i]) {
                minValue = val[i];
            }
            if (maxValue < val[i]) {
                maxValue = val[i];
            }
        }
        for (std::size_t i = 0; i < val.size(); i++) {
            val[i] = (val[i] - minValue)/(maxValue - minValue);
        }
        return;
    }
    static void set3D(Tensor_ &y, const Tensor_ &x, int offsetR, int offsetC)
    {
        /* (c, h, w) */
        assert(y.shape[0] == x.shape[0]);
        assert(y.shape[1] > x.shape[1] && y.shape[2] > x.shape[2]);
        for (int i = 0; i < x.shape[0]; i++) {
            for (int h = 0; h < x.shape[1]; h++) {
                for (int k = 0; k < x.shape[2]; k++) {
                    y(i, h + offsetR, k + offsetC) = x(i, h, k);
                }
            }
        }
        return;
    }

    static void copy(Tensor_ &x, const Tensor_ &x_)
    {
        for (std::size_t i = 0; i < x.totalSize; i++) {
            x.val[i] = x_.val[i];
        }
        return;
    }
    static void foreach(Tensor_ &y, const Tensor_ &x, std::function<T(T)> func_)
    {
        for (std::size_t i = 0; i < y.totalSize; i++) {
            y.val[i] = func_(x.val[i]);
        }
        return;
    }

    static Tensor_ func(const Tensor_ &x, std::function<T(T)> func_)
    {
        Tensor_ y(x.shape);
        for (std::size_t i = 0; i < y.totalSize; i++) {
            y.val[i] = func_(x.val[i]);
        }
        return y;
    }
    static void func(Tensor_ &y, const Tensor_ &x, std::function<T(T)> func_)
    {
        for (std::size_t i = 0; i < y.totalSize; i++) {
            y.val[i] = func_(x.val[i]);
        }
        return y;
    }
    static Tensor_ func(const Tensor_ &x, T(*func_)(T))
    {
        Tensor_ y(x.shape);
        for (std::size_t i = 0; i < y.totalSize; i++) {
            y.val[i] = func_(x.val[i]);
        }
        return y;
    }
    /* matrix operation */
    struct MatOp {
        static void ikkj(Tensor_ &x, const Tensor_ &x1, const Tensor_ &x2)
        {
            for (std::size_t i = 0; i < x.shape[0]; i++) {
                for (std::size_t j = 0; j < x.shape[1]; j++) {
                    for (std::size_t k = 0; k < x1.shape[1]; k++) {
                        /* (i, j) = (i, k) * (k, j) */
                        x.val[i*x.shape[1] + j] += x1.val[i*x1.shape[1] + k]*x2.val[k*x2.shape[1] + j];
                    }
                }
            }
            return;
        }
        static void kikj(Tensor_ &x, const Tensor_ &x1, const Tensor_ &x2)
        {
            /* transpose x1 */
            for (std::size_t i = 0; i < x.shape[0]; i++) {
                for (std::size_t j = 0; j < x.shape[1]; j++) {
                    for (std::size_t k = 0; k < x1.shape[0]; k++) {
                        /* (i, j) = (k, i)^T * (k, j)^T */
                        x.val[i*x.shape[1] + j] += x1.val[k*x1.shape[1] + i]*x2.val[k*x2.shape[1] + j];
                    }
                }
            }
            return;
        }
        static void ikjk(Tensor_ &x, const Tensor_ &x1, const Tensor_ &x2)
        {
            /* transpose x2 */
            for (std::size_t i = 0; i < x.shape[0]; i++) {
                for (std::size_t j = 0; j < x.shape[1]; j++) {
                    for (std::size_t k = 0; k < x1.shape[1]; k++) {
                        /* (i, j) = (i, k) * (j, k)^T */
                        x.val[i*x.shape[1] + j] += x1.val[i*x1.shape[1] + k]*x2.val[j*x2.shape[1] + k];
                    }
                }
            }
            return;
        }
        static void kijk(Tensor_ &x, const Tensor_ &x1, const Tensor_ &x2)
        {
            /* transpose x1, x2 */
            for (std::size_t i = 0; i < x.shape[0]; i++) {
                for (std::size_t j = 0; j < x.shape[1]; j++) {
                    for (std::size_t k = 0; k < x1.shape[0]; k++) {
                        /* (i, j) = (k, i)^T * (j, k)^T */
                        x.val[i*x.shape[1] + j] += x1.val[k*x1.shape[1] + i] * x2.val[j*x2.shape[1] + k];
                    }
                }
            }
            return;
        }

        static void set(Tensor_ &y, const Tensor_ &x, int offsetR, int offsetC)
        {
            /* (h, w) */
            for (int h = 0; h < x.shape[0]; h++) {
                for (int k = 0; k < x.shape[1]; k++) {
                    y(h + offsetR, k + offsetC) = x(h, k);
                }
            }
            return;
        }

        static void setRow(Tensor_ &y, size_t i, const Tensor_ &row)
        {
            /*
                y: (r, c)
                row: (1, c)
            */
            for (std::size_t j = 0; j < row.shape[1]; j++) {
                y(i, j) = row(0, j);
            }
            return;
        }

        static void setColumn(Tensor_ &y, size_t j, const Tensor_ &col)
        {
            /*
                y: (r, c)
                col: (r, 1)
            */
            for (std::size_t i = 0; i < col.shape[0]; i++) {
                y(i, j) = col(i, 0);
            }
            return;
        }

        static void row(const Tensor_ &x, size_t i, Tensor_ &row)
        {
            /*
                x: (r, c)
                row: (1, c)
            */
            for (std::size_t j = 0; j < row.shape[1]; j++) {
                row(0, j) = x(i, j);
            }
            return;
        }

        static void column(const Tensor_ &x, size_t j, Tensor_ &col)
        {
            /*
                y: (r, c)
                col: (r, 1)
            */
            for (std::size_t i = 0; i < col.shape[0]; i++) {
                col(i, 0) = x(i, j);
            }
            return;
        }
    };
    /* tensor product */
    static Tensor_ product2D(const Tensor_& x1, const Tensor_& x2)
    {
        int r = x1.shape[0]*x2.shape[0];
        int c = x1.shape[1]*x2.shape[1];
        Tensor_ y(r, c);
        for (int i = 0; i < x1.shape[0]; i++) {
            for (int j = 0; j < x1.shape[1]; j++) {
                for (int h = 0; h < x2.shape[0]; h++) {
                    for (int k = 0; k < x2.shape[1]; k++) {
                        y(h + i*x1.shape[0], k + j*x1.shape[1]) = x1(i, j)*x2(h, k);
                    }
                }
            }
        }
        return y;
    }

    static void product2D(Tensor_& y, const Tensor_& x1, const Tensor_& x2)
    {
        for (int i = 0; i < x1.shape[0]; i++) {
            for (int j = 0; j < x1.shape[1]; j++) {
                for (int h = 0; h < x2.shape[0]; h++) {
                    for (int k = 0; k < x2.shape[1]; k++) {
                        y(h + i*x1.shape[0], k + j*x1.shape[1]) = x1(i, j)*x2(h, k);
                    }
                }
            }
        }
        return y;
    }

    /* display */
    void printValue() const
    {
        std::cout<<"[";
        for (std::size_t i = 0; i < val.size(); i++) {
            std::cout<<val[i];
            if (i < totalSize - 1) {
                std::cout<<",";
            }
        }
        std::cout<<"]"<<std::endl;
        return;
    }

    void printShape() const
    {
        std::cout<<"(";
        for (std::size_t i = 0; i < shape.size(); i++) {
            std::cout<<shape[i]<<",";
        }
        std::cout<<")"<<std::endl;
        return;
    }

};


using Tensori = Tensor_<int>;
using Tensor = Tensor_<float>;
#endif // TENSOR_H
