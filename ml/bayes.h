#ifndef BAYES_H
#define BAYES_H
#include <map>
#include <vector>
#include <string>
#include "../basic/vec.h"
#include "../Statistics/dataset.h"

/*
    1. NaiveBayes
        P = prior*likelihood/evidence
        P(A|B) = P(A)*P(B|A)/P(B)
        P(A|B1,B2,...,Bn) = P(A)*P(B1,B2,...,Bn|A)/P(B1,B2,...,Bn)
    2. Conditional Indepdence
        P(A|B1,B2,...,Bn) = P(A)*∏ P(Bi|A)/∏ P(Bi)
        P(A|B1,B2,...,Bn) ≈ P(A)*∏ P(Bi|A)
    3. classify
        classify(b1,b2,...,bn) = argmax P(A=a)*∏ P(Bi=bi|A=a)

        P(B=bk|A=a) = P(A=a|B=bk)*P(B=bk)/Σ P(A=a|B=bi)*P(B=bi)
        f = P(B=bk|A=a) = P(B=bk)*∏ P(A=aj|B=bk)/Σ P(B=bi)*P(A=aj|B=bi)
        f = argmax P(B=bk)*∏ P(A=aj|B=bk)

        P(A) = Σ I(A=ai)/N
        P(A=ai|B=b) = Σ I(A=ai, B=b)/Σ I(B=b)

    4. Bernoulli distribution
        x = [0, 1, 1, 0, 0], y = [0]
        x = [0, 1, 0, 1, 0], y = [1]
        x = [1, 0, 1, 0, 0], y = [0]
        x = [1, 0, 0, 1, 1], y = [1]
        featureDim = 2

        P(x=[0, 1, 1, 0, 0]|y=0) = 1/4
        P(x=[1, 0, 1, 0, 0]|y=0) = 1/4
        P(x=[0, 1, 0, 1, 0]|y=1) = 1/4
        P(x=[1, 0, 0, 1, 1]|y=1) = 1/4

*/

class NaiveBayes
{
public:
    float lambda;
    std::size_t featureDim;
    std::size_t outputDim;
    /* (outputDim, 1) */
    Vec prior;
    /* (outputDim, featureDim) */
    std::vector<Vec> condit;
public:
    NaiveBayes():lambda(1){}

    std::size_t estimatePosterior(const Vec &x, Vec &p)
    {
        /*
           y = argmax P(Y=yk)*∏ P(X=xi|Y=yk)
           y = argmax log(P(Y=yk)) + Σ log(P(X=xi|Y=yk))
        */
        for (std::size_t i = 0; i < condit.size(); i++) {
            p[i] += std::log(prior[i]);
            for (std::size_t j = 0; j < condit[0].size(); j++) {
                if (x[j] != 0) {
                    p[i] += std::log(condit[i][j]);
                }
            }
        }
        return p.argmax();
    }

    void count(const std::vector<Vec> &x, const Vec &y, std::vector<Vec> &featureCount, Vec &labelCount)
    {
        /* featureCount: (label, feature) */
        for (std::size_t k = 0; k < outputDim; k++) {
            for (std::size_t i = 0; i < x.size(); i++) {
                for (std::size_t j = 0; j < featureDim; j++) {
                    featureCount[k][j] += x[i][j];
                }
                labelCount[k]++;
            }
        }
        return;
    }
    void estimateCondition(const std::vector<Vec> &x, const Vec &y)
    {
        /* construct */
        prior = Vec(outputDim, 1);
        condit = std::vector<Vec>(outputDim, Vec(featureDim, 1));
        Vec labelCount = Vec(outputDim, 0);
        std::vector<Vec> featureCount = std::vector<Vec>(outputDim, Vec(featureDim, 0));
        /* count */
        count(x, y, featureCount, labelCount);
        /* prior probability */
        for (std::size_t i = 0; i < prior.size(); i++) {
            /* P(Y=ck) = Σ I(yi=ck)/N */
            prior[i] = float(labelCount[i] + lambda)/float(y.size() + lambda);
        }
        /* conditional probability */
        for (std::size_t i = 0; i < condit.size(); i++) {
            for (std::size_t j = 0; j < condit[0].size(); j++) {
                /* P(X=ak|Y=ck) = Σ I(xi=ak, yi=ck)/Σ I(yi=ck) */
                condit[i][j] = float(featureCount[i][j] + lambda)/float(labelCount[i] + lambda);
            }
        }
        return;
    }
};


class GaussianBayes
{
public:
    std::vector<Vec> u;
    std::vector<Vec> sigma;
    Vec prior;
    std::size_t outputDim;
    std::size_t featureDim;
public:
    GaussianBayes()
    {
        u = std::vector<Vec>(outputDim, Vec(featureDim, 0));
        sigma = std::vector<Vec>(outputDim, Vec(featureDim, 0));
        prior = Vec(outputDim, 0);
    }

    void estimate(const std::vector<Vec> &x, const Vec &y)
    {
        /* count */
        Vec labelCount(outputDim, 0);
        /* prior */
        for (std::size_t i = 0; i < outputDim; i++) {
            prior[i] /= float(y.size());
        }
        /* mean */
        for (std::size_t i = 0; i < outputDim; i++) {
            for (std::size_t j = 0; j < x.size(); j++) {
                if (y[i] == float(i)) {
                    u[i] += x[j];
                    labelCount[i]++;
                }
            }
        }
        for (std::size_t i = 0; i < outputDim; i++) {
            u[i] /= labelCount;
        }
        /* var */
        for (std::size_t i = 0; i < outputDim; i++) {
            for (std::size_t j = 0; j < x.size(); j++) {
                for (std::size_t k = 0; k < x[0].size(); j++) {
                    if (y[i] == float(i)) {
                        sigma[i][k] += (x[j][k] - u[i][k])*(x[j][k] - u[i][k]);
                    }
                }
            }
        }
        return;
    }

    std::size_t operator()(const Vec &x)
    {
        /*
            y = argmax log(P(yc)) - 0.5*Σ log(2*pi*sigma^2) - 0.5*Σlog((xi-ui)^2/sigma^2))
        */
        Vec p(outputDim, 0);
        for (std::size_t i = 0; i < outputDim; i++) {
            p[i] = std::log(prior[i]);
            for (std::size_t j = 0; j < featureDim; j++) {
                p[i] -= 0.5*(std::log(2*3.14*sigma[i][j]) + std::log((x[j] - u[i][j])*(x[j] - u[i][j])/sigma[i][j]));
            }
        }
        return p.argmax();
    }
};

class TextBayes : public NaiveBayes
{
public:
    std::map<std::size_t, std::vector<std::string> > keywords;
    std::vector<std::string> labels;
    std::vector<std::string> banishWords;
public:
    TextBayes(const std::string &keywordFile)
    {
        /*
            <key1>,<key2>,...<keyn>
            value1,value2,...valuen
        */
        CSV<Strings> db(keywordFile);
        std::string keyword;
        std::vector<Strings> data;
        int ret = db.load(keyword, data);
        if (ret < 0) {
            return;
        }
        /* labels */
        Text::split(keyword, labels, ",");
        /* keywords */
        for (std::size_t i = 0; i < labels.size(); i++) {
            for (std::size_t j = 0; j < data.size(); j++) {
                keywords[i].push_back(data[j][i]);
            }
        }
    }

    std::string operator()(const std::string &text)
    {
        /* word to vector */
        Vec x;
        Text::toVector(labels, text, x);

        /* calculate posterior */
        Vec p(x.size(), 0);
        NaiveBayes::estimatePosterior(x, p);
        std::size_t index = p.argmax();
        return labels[index];
    }

    void estimateCondition(const std::string &text)
    {
        /* text to words */

        /* word frequence */

        /* remove banished word */

        /* text feature */

    }
};

#endif // BAYES_H
