#ifndef DATASET_H
#define DATASET_H
#include <vector>
#include <set>
#include <memory>
#include <string>
#include "csv.h"
#include "../basic/vec.h"
#include "../basic/mat.h"
#include "../basic/tensor.hpp"

class NumericDB : public CSV<Vec>
{
public:
    NumericDB(const std::string &fileName)
        :CSV<Vec>(fileName){}
    int load(std::vector<Vec> &data)
    {
        std::string title;
        return CSV<Vec>::load(title, data);
    }
    int load(std::vector<Mat> &x)
    {
        std::string title;
        std::vector<Vec> data;
        int ret = CSV<Vec>::load(title, data);
        if (ret < 0) {
            return -1;
        }
        for (std::size_t i = 0; i < data.size(); i++) {
            x.push_back(Mat(1, cols, data[i]));
        }
        return 0;
    }
    int load(std::vector<Tensor> &x)
    {
        std::string title;
        std::vector<Vec> data;
        int ret = CSV<Vec>::load(title, data);
        if (ret < 0) {
            return -1;
        }
        for (std::size_t i = 0; i < data.size(); i++) {
            x.push_back(Tensor({1, int(cols)}, data[i]));
        }
        return 0;
    }

};


class Text
{
public:
    static void split(const std::string &sentence, std::vector<std::string> &words, const std::string &pattern)
    {
        std::string sentence_(sentence);
        sentence_ += pattern;
        for (std::size_t i = 0; i < sentence_.size(); i++) {
            int pos = sentence_.find(pattern, i);
            if (pos < 0) {
                continue;
            }
            std::string term = sentence_.substr(i, pos - i);
            words.push_back(term);
            i = pos + pattern.size() - 1;
        }
        return;
    }

    static void getSentence(const std::string &text, std::vector<std::string> &sentences)
    {
        split(text, sentences, ".");
        return;
    }

    static std::string removeSymbol(const std::string &text)
    {
        /* remove , ' " . */
        std::string s(text);
        for (std::size_t i = 0; i < s.size(); i++) {
            if (s[i] == ',' || s[i] == '"' || s[i] == '\'' || s[i] == '.') {
                s[i] = ' ';
            }
        }
        return s;
    }

    static void toVector(const std::vector<std::string> &eigens, const std::string &text, Vec &x)
    {
        /* sentence to single words */
        std::string s = removeSymbol(text);
        std::vector<std::string> words;
        split(text, words, " ");
        /* filter */
        std::set<std::string> filter(words.begin(), words.end());
        std::vector<std::string> wordTuple(filter.begin(), filter.end());
        /* word to vector */
        x = Vec(eigens.size(), 0);
        for (std::size_t i = 0; i < eigens.size(); i++) {
            for (std::size_t j = 0; j < wordTuple.size(); j++) {
                if (eigens[i] == wordTuple[j]) {
                    x[i] = 1;
                }
            }
        }
        return;
    }

    static void wordFrequence(const std::string &text, std::vector<std::string> &wordTuple, Vec &x)
    {
        /* sentence to single words */
        std::string s = removeSymbol(text);
        std::vector<std::string> words;
        split(text, words, " ");
        /* filter */
        std::set<std::string> filter(words.begin(), words.end());
        wordTuple.assign(filter.begin(), filter.end());
        /* statistics */
        x = Vec(wordTuple.size(), 0);
        for (std::size_t i = 0; i < wordTuple.size(); i++) {
            for (std::size_t j = 0; j < words.size(); j++) {
                if (words[j] == wordTuple[i]) {
                    x[i]++;
                }
            }
        }
        return;
    }
};

class BinaryLoader
{
public:
    static std::unique_ptr<uint8_t> load(const std::string &fileName)
    {
        if (fileName.empty()) {
            return nullptr;
        }
        std::ifstream file(fileName, std::ios::binary|std::ios::ate);
        std::streamsize totalSize = file.tellg();
        file.seekg(0, std::ios::beg);
        if (totalSize == -1) {
            return nullptr;
        }
        std::unique_ptr<uint8_t> buffer(new uint8_t[totalSize]);
        file.read((char*)buffer.get(), totalSize);
        file.close();
        return buffer;
    }

    inline static uint32_t byteswap(uint32_t a)
    {
        return ((((a >> 24) & 0xff) << 0) |
            (((a >> 16) & 0xff) << 8) |
            (((a >> 8) & 0xff) << 16) |
            (((a >> 0) & 0xff) << 24));
    }

};


class MnistLoader
{
public:
    std::size_t N;
    std::string dataPath;
    std::string labelPath;
    std::vector<Tensor> x;
    std::vector<Tensor> yt;
public:
    explicit MnistLoader(const std::string &dataPath_, const std::string &labelPath_)
        :dataPath(dataPath_),labelPath(labelPath_){}

    int load()
    {
        /* load data */
        std::unique_ptr<uint8_t> datas = BinaryLoader::load(dataPath);
        if (datas == nullptr) {
            std::cout<<"load training data failed."<<std::endl;
            return -1;
        }
        N = BinaryLoader::byteswap(*(uint32_t*)(datas.get() + 4));
        x = std::vector<Tensor> (N, Tensor(1, 28, 28));
        for (std::size_t n = 0; n < N; n++ ) {
            uint8_t* img = datas.get() + 16 + n*(28*28);
            for (std::size_t i = 0; i < x[n].totalSize; i++ ) {
                x[n][i] = img[i]/255.0f;
            }
        }
        /* load label */
        std::unique_ptr<uint8_t> labels = BinaryLoader::load(labelPath);
        if (labels == nullptr) {
            std::cout<<"load training label failed."<<std::endl;
            return -1;
        }
        yt = std::vector<Tensor>(N, Tensor(10, 1));
        for (std::size_t n = 0; n < N; n++ ) {
            uint8_t* label = labels.get() + 8 + n;
            yt[n](*label, 0) = 1.0f;
        }
        return 0;
    }
};

#endif // DATASET_H
