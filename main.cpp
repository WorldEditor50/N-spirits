#include <iostream>
#include "csv.h"
#include "dataset.h"
#include "kmeans.h"
#include "svm.h"
#include "gmm.h"
#include "mat.h"
#include "linearalgebra.h"
#include "utils.h"


class Student
{
public:
    std::string name;
    int age;
    float score;
public:
    Student(){}
    Student(const std::string &name_, int age_, int score_)
        :name(name_),age(age_),score(score_){}
    static void parse(std::istringstream &stream, std::size_t cols, Student &data)
    {
        std::getline(stream, data.name, ',');
        std::string age;
        std::getline(stream, age, ',');
        data.age = std::atoi(age.c_str());
        std::string score;
        std::getline(stream, score, ',');
        data.score = std::atof(score.c_str());
        return;
    }

    static void toString(const Student &data, std::string &line)
    {
        line = data.name + "," + std::to_string(data.age) + "," + std::to_string(data.score);
        return;
    }

};

void test_csv()
{

    CSV<Student> db("stu.csv");

    db.save("name,age,score", {Student("Tom", 12, 90.0),
                               Student("Jim", 15, 88.0),
                               Student("John", 10, 70.0),
                               Student("Jenny", 17, 80.0) });

    std::string line;
    Student stu;
    db.read(2, stu);
    std::cout<<stu.name<<std::endl;

    std::vector<Student> students;
    std::string title;
    db.load(title, students);
    for (std::size_t i = 0; i < students.size(); i++) {
        std::cout<<students[i].name<<","<<students[i].age<<","<<students[i].score<<std::endl;
    }
    return;
}

void test_lu()
{
    Mat x(3, 3, { 1, 1, 1,
                  0, 0.5, -2,
                  0, 1, 1});
    Mat xi;
    LinearAlgebra::LU::inv(x, xi);
    xi.show();
    std::cout<<"test inverse:"<<std::endl;
    Mat I(x.rows, x.cols);
    Mat::Multiply::ikkj(I, x, xi);
    I.show();
    return;
}

void test_det()
{
    float value;
    Mat x1(3, 3, {1, 1, 1,
                  1, 2, 3,
                  1, 5, 1});
    LinearAlgebra::det(x1, value);
    std::cout<<"det:"<<value<<std::endl;



    Mat x2(4, 4, {1, 1, 1, 2,
                  1, 2, 3, 0,
                  0, 5, 1, -1,
                  1, 0, -3, 1});
    LinearAlgebra::det(x2, value);
    std::cout<<"det:"<<value<<std::endl;
    return;
}

void test_qr()
{
    Mat x(3, 3, {1, 1, 1,
                 1, 2, 3,
                 1, 5, 1});
    Mat q;
    Mat r;
    LinearAlgebra::QR::solve(x, q, r);
    std::cout<<"Q:"<<std::endl;
    q.show();
    std::cout<<"R:"<<std::endl;
    r.show();
    return;
}

void test_svd()
{
    Mat x1(3, 3, {1, 1, 1,
                 1, 2, 3,
                 1, 5, 1});

    Mat x(10, 5, {0.0162, 0.878, 0.263, 0.0955, 0.359,
                  0.329, 0.326, 0.757, 0.165, 0.728,
                  0.609, 0.515, 0.385, 0.908, 0.89,
                  0.91, 0.06, 0.43, 0.691, 0.96,
                  0.476, 0.0498, 0.65, 0.378, 0.672,
                  0.914, 0.788, 0.285, 0.447, 0.0846,
                  0.495, 0.463, 0.962, 0.758, 0.558,
                  0.321, 0.0872, 0.884, 0.0788, 0.252,
                  0.612, 0.688, 0.767, 0.997, 0.597,
                  0.657, 0.907, 0.657, 0.0873, 0.598
          });
    Mat u;
    Mat v;
    Mat s;
    LinearAlgebra::SVD::solve(x, u, s, v);
    std::cout<<"U:"<<std::endl;
    u.show();
    std::cout<<"S:"<<std::endl;
    s.show();
    std::cout<<"V:"<<std::endl;
    v.show();
    /* x = U*S*V^T */
    Mat y = u*s*v.tr();
    std::cout<<"x:"<<std::endl;
    x.show();
    std::cout<<"y:"<<std::endl;
    y.show();
    return;
}

void test_transpose_multiply()
{
    Mat x1(2, 3, {1, 2, 3,
                  4, 5, 6});
    Mat x2(3, 2, {1, 2,
                  3, 4,
                  5, 6});
    Mat x3(2, 3, {1, 0, 3,
                  4, 2, 0});
    Mat x4(3, 2, {0, 2,
                  3, 4,
                  1, 0});
    std::cout<<"ikkj:"<<std::endl;
    Mat x5(2, 2);
    /* (2, 3) * (3, 2) = (2, 2) */
    Mat::Multiply::ikkj(x5, x1, x2);
    /* [[22, 28], [49, 64]] */
    x5.show();
    std::cout<<"kikj:"<<std::endl;
    Mat x6(2, 2);
    /* (3, 2) * (3, 2) = (2, 2) */
    Mat::Multiply::kikj(x6, x4, x2);
    /*
        0, 3, 1     1, 2
        2, 4, 0     3, 4
                    5, 6
        [[14, 18], [14, 20]]
    */
    x6.show();
    std::cout<<"ikjk:"<<std::endl;
    Mat x7(2, 2);
    /* (2, 3) * (2, 3) = (2, 2) */
    Mat::Multiply::ikjk(x7, x1, x3);
    /*
        1, 2, 3   1, 4
        4, 5, 6   0, 2
                  3, 0

        [[10, 8], [22, 26]]
    */
    x7.show();
    std::cout<<"kijk:"<<std::endl;
    Mat x8(2, 2);
    /* (3, 2) * (2, 3) = (2, 2) */
    Mat::Multiply::kijk(x8, x4, x3);
    /*
        0, 2      1, 0, 3
        3, 4      4, 2, 0
        1, 0

        0, 3, 1     1, 4
        2, 4, 0     0, 2
                    3, 0

        [[3, 6], [2, 16]]
    */
    x8.show();
    /* rand */
    Mat x9(3, 4);
    Mat x10(3, 5);
    Mat x11(4, 5);
    Utils::uniform(x9, 0, 9);
    Utils::uniform(x10, 0, 9);
    Mat::Multiply::kikj(x11, x9, x10);
    std::cout<<"x9:"<<std::endl;
    x9.show();
    std::cout<<"x10:"<<std::endl;
    x10.show();
    std::cout<<"x11:"<<std::endl;
    x11.show();
    return;
}

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
    return;
}

int main()
{
#if 0
    test_lu();
    test_qr();
    test_det();
    test_svd();
#endif
    test_kmeans();
    return 0;
}
