#ifndef DATASET_H
#define DATASET_H
#include <vector>
#include <set>
#include "csv.h"
#include "../basic/vec.h"
#include "../basic/mat.h"

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

#endif // DATASET_H
