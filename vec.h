#ifndef VEC_H
#define VEC_H
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

class Vec : public std::vector<float>
{
public:
    Vec(){}
    explicit Vec(std::size_t s, float val):std::vector<float>(s, val){}
    explicit Vec(const std::vector<float> &x):std::vector<float>(x){}
    explicit Vec(const std::vector<float>::iterator &itBegin, const std::vector<float>::iterator &itEnd)
        :std::vector<float>(itBegin, itEnd){}
    Vec& operator /=(float val)
    {
        float* x = data();
        for (std::size_t i = 0; i < this->size(); i++) {
            x[i] /= val;
        }
        return *this;
    }
    Vec& operator *=(float val)
    {
        float* x = data();
        for (std::size_t i = 0; i < this->size(); i++) {
            x[i] *= val;
        }
        return *this;
    }
    Vec& operator +=(float val)
    {
        float* x = data();
        for (std::size_t i = 0; i < this->size(); i++) {
            x[i] += val;
        }
        return *this;
    }
    Vec& operator -=(float val)
    {
        float* x = data();
        for (std::size_t i = 0; i < this->size(); i++) {
            x[i] -= val;
        }
        return *this;
    }
    Vec& operator +=(const Vec &y)
    {
        float* x = data();
        for (std::size_t i = 0; i < this->size(); i++) {
            x[i] += y[i];
        }
        return *this;
    }
    Vec& operator -=(const Vec &y)
    {
        float* x = data();
        for (std::size_t i = 0; i < this->size(); i++) {
            x[i] -= y[i];
        }
        return *this;
    }
    Vec& operator *=(const Vec &y)
    {
        float* x = data();
        for (std::size_t i = 0; i < this->size(); i++) {
            x[i] *= y[i];
        }
        return *this;
    }
    Vec& operator /=(const Vec &y)
    {
        float* x = data();
        for (std::size_t i = 0; i < this->size(); i++) {
            x[i] /= y[i];
        }
        return *this;
    }

    void zero()
    {
        float* x = data();
        for (std::size_t i = 0; i < this->size(); i++) {
            x[i] = 0;
        }
        return;
    }
    std::size_t argmax() const
    {
        const float* x = data();
        float val = x[0];
        std::size_t index = 0;
        for (std::size_t i = 1; i < this->size(); i++) {
            if (x[i] > val) {
                val = x[i];
                index = i;
            }
        }
        return index;
    }
    std::size_t argmin() const
    {
        const float* x = data();
        float val = x[0];
        std::size_t index = 0;
        for (std::size_t i = 1; i < this->size(); i++) {
            if (x[i] < val) {
                val = x[i];
                index = i;
            }
        }
        return index;
    }
    float max() const
    {
        const float* x = data();
        float val = x[0];
        for (std::size_t i = 1; i < this->size(); i++) {
            if (x[i] > val) {
                val = x[i];
            }
        }
        return val;
    }
    float min() const
    {
        const float* x = data();
        float val = x[0];
        for (std::size_t i = 1; i < this->size(); i++) {
            if (x[i] < val) {
                val = x[i];
            }
        }
        return val;
    }
    float sum() const
    {
        const float* x = data();
        float val = 0;
        for (std::size_t i = 0; i < this->size(); i++) {
            val += x[i];
        }
        return val;
    }
    static float dot(const Vec &x1, const Vec &x2)
    {
        float s = 0;
        for (std::size_t i = 1; i < x1.size(); i++) {
            s += x1[i]*x2[i];
        }
        return s;
    }
    static void multi(Vec &y, const Vec &x1, const Vec &x2)
    {
        for (std::size_t i = 1; i < x1.size(); i++) {
            y[i] = x1[i]*x2[i];
        }
        return;
    }
    static void parse(std::istringstream &stream, std::size_t cols, Vec &data)
    {
        for (std::size_t i = 0; i < cols; i++) {
            std::string x;
            std::getline(stream, x, ',');
            data.push_back(std::atof(x.c_str()));
        }
        return;
    }

    static void toString(const Vec &data, std::string &line)
    {
        for (std::size_t i = 0; i < data.size(); i++) {
            line += std::to_string(data[i]);
            if (i < data.size() - 1) {
                line += ",";
            }
        }
        return;
    }

    void show() const
    {
        const float* x = data();
        for (std::size_t i = 0; i < this->size(); i++) {
            std::cout<< x[i] <<" ";
        }
        std::cout<<std::endl;
        return;
    }
};
#endif // VEC_H
