#include <iostream>
#include "../basic/linearalgebra.h"
#include "../ml/kmeans.h"
#include "../ml/svm.h"
#include "../ml/gmm.h"
#include "../ml/kdtree.hpp"
#include "../utils/dataset.h"

void test_kmeans()
{
    /* load data */
    NumericDB db("D:/home/dataset/wine-clustering.csv");
    std::vector<Mat> x;
    db.load(x);
    /* clustering */
    KMeans model(3);
    model.cluster(x, 1000);
    /* predict */
    std::size_t label = model(x[0]);
    std::cout<<"label:"<<label<<std::endl;
    /* project to 2d-plane */
    LinearAlgebra::PCA pca;
    Mat x1;
    Mat::fromArray(x, x1);
    pca.fit(x1);
    Mat y;
    pca.project(x[0], 2, y);
    y.show();
    return;
}

void test_kdtree()
{
    std::vector<Tensor> x = {Tensor({4, 1}, {1, 1, 2, 3}),
                             Tensor({4, 1}, {0, 5, 4, 0}),
                             Tensor({4, 1}, {1, -1, 0, 1}),
                             Tensor({4, 1}, {2, 1, 3, -5}),
                             Tensor({4, 1}, {-1, 0, 1, 1}),
                             Tensor({4, 1}, {9, 2, 7, 8}),
                             Tensor({4, 1}, {7, 9, -1, 2}),
                             Tensor({4, 1}, {2, -4, 9, 0}),
                             Tensor({4, 1}, {4, 1, 0, 3}),
                             Tensor({4, 1}, {8, 2, 1, 1}),
                             Tensor({4, 1}, {0, 1, 1, 1}),
                             Tensor({4, 1}, {0, 0, 0, 1})};

    Tensor u = Statistics::sum(x) / x.size();
    u.printValue();
    Tensor sigma = Statistics::variance(x, u);
    sigma.printValue();
    //return 0;
    KDTree kdtree(x);
    //kdtree.display();
    //return 0;
    std::vector<KDTree::Result> results;
    Tensor xi({4, 1}, {0, 0, 0, 0});
    kdtree.find(xi, results);
    for (std::size_t i = 0; i < results.size(); i++) {
        results[i].value.printValue();
        std::cout<<"d = "<<results[i].distance<<std::endl;
    }
    return;
}

int main()
{
    //test_kmeans();
    test_kdtree();
	return 0;
}
