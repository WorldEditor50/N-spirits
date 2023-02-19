#include <iostream>
#include <tuple>
#include "basic/mat.h"
#include "basic/linearalgebra.h"
#include "basic/tensor.h"
#include "basic/complexnumber.h"
#include "basic/utils.h"
#include "utils/csv.h"
#include "utils/dataset.h"
#include "kmeans.h"
#include "svm.h"
#include "gmm.h"


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
    y.show();
    return;
}


void test_tensor()
{
    Tensori x(2, 3, 3);
    for (std::size_t i = 0; i < x.sizes.size(); i++) {
        std::cout<<x.sizes[i]<<",";
    }
    std::cout<<std::endl;
    for (std::size_t i = 0; i < x.shape.size(); i++) {
        std::cout<<x.shape[i]<<",";
    }
    std::cout<<std::endl;

    x.val = std::vector<int>{
                                1, 2, 3,
                                4, 5, 6,
                                7, 8, 9,

                                11, 12, 13,
                                14, 15, 16,
                                17, 18, 19
                             };

    std::cout<<x(1, 1, 2)<<std::endl;
    x.printShape();

    Tensori x1 = x.sub(0);
    x1.printValue();

    x.reshape(3, 3, 2);
    Tensori x2 = x.sub(0, 1);
    x2.printValue();
    return;
}


void conv(Tensori &o, const Tensori &kernels, const Tensori &x, int stride=1, int padding=1)
{
    for (int n = 0; n < o.shape[0]; n++) {
        for (int i = 0; i < o.shape[1]; i++) {
            for (int j = 0; j < o.shape[2]; j++) {
                /* kernels */
                for (int h = 0; h < kernels.shape[2]; h++) {
                    for (int k = 0; k < kernels.shape[3]; k++) {
                        for (int c = 0; c < kernels.shape[1]; c++) {
                            /* map to input  */
                            int row = h + i*stride - padding;
                            int col = k + j*stride - padding;
                            if (row < 0 || row >= x.shape[1] ||
                                    col < 0 || col >= x.shape[2]) {
                                continue;
                            }
                            //std::cout<<"(h, k)=("<<h<<","<<k<<"), "<<"(row, col)=("<<row<<","<<col<<")"<<std::endl;
                            /* sum up all input channel's convolution result */
                            o(n, i, j) += kernels(n, c, h, k)*x(c, row, col);
                        }
                    }
                }
                //std::cout<<"-----------------------------------"<<std::endl;
            }
        }
    }
    return;
}

void test_conv()
{

    /*
            0   0   0   0   0

            0   1   2   3   0        1     0     1           4     8     2

            0   4   5   6   0   *    0    -1     0      =    6     15    4

            0   7   8   9   0        1     0     1          -2     2    -4

            0   0   0   0   0



            0   0   0   0   0

            0   1   1   1   0        1     1     1           4     6     4

            0   1   1   1   0   *    1     1     1      =    6     9     6

            0   1   1   1   0        1     1     1           4     6     4

            0   0   0   0   0



            1   1   1        1     1           4     4

            1   1   1   *    1     1      =    4     4

            1   1   1


            0   0   0   0   0   0   0

            0   0   0   0   0   0   0                                 1    2     3     2    1

            0   0   1   1   1   0   0           1     1     1         2    4     6     4    2

            0   0   1   1   1   0   0      *    1     1     1      =  3    6     9     6    3

            0   0   1   1   1   0   0           1     1     1         2    4     6     4    2

            0   0   0   0   0   0   0                                 1    2     3     2    1

            0   0   0   0   0   0   0


    */

    Tensori o(1, 3, 3);
    Tensori kernels({1, 1, 3, 3}, {1, 0, 1, 0, -1, 0, 1, 0, 1});
    Tensori x({1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    /* case 1: */
    conv(o, kernels, x);
    o.printValue();
    /* case 2: */
    o.zero();
    kernels.zero();
    x.zero();

    kernels.assign(1, 1, 1, 1, 1, 1, 1, 1, 1);
    x.assign(1, 1, 1, 1, 1, 1, 1, 1, 1);

    conv(o, kernels, x);
    o.printValue();
    /* case 3: */
    Tensori y1(1, 2, 2);
    Tensori x1({1, 3, 3}, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    Tensori k1({1, 1, 2, 2}, {1, 1, 1, 1});
    conv(y1, k1, x1, 1, 0);
    y1.printValue();
    /* case 4: */
    Tensori y2(1, 5, 5);
    Tensori x2({1, 3, 3}, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    Tensori k2({1, 1, 3, 3}, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    conv(y2, k2, x2, 1, 2);
    y2.printValue();
    /* case 5: */
    Tensori y3(1, 3, 3);
    conv(y3, k2, x2, 2, 2);
    y3.printValue();
    return;
}

void test_tensor_func()
{
    Tensor x({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor y = Tensor::func(x, [&](float xi)->float{ return xi*xi;});
    y.printValue();
    y = Tensor::func(x, exp);
    y.printValue();
    return;
}

void test_product()
{
    Tensor x1({2, 2}, {1, 2, 3, 4});
    Tensor x2({2, 2}, {9, 9, 9, 9});
    Tensor y = Tensor::product2D(x1, x2);
    y.printValue();
    return;
}

void test_permute()
{
    Tensor x({2, 4, 3}, {
                 1, 2, 3,
                 1, 4, 5,
                 6, 0, 7,
                 8, 9, 2,

                 10, 11, 12,
                 13, 14, 15,
                 16, 17, 18,
                 19, 20, 21
             });
    /*
       permute:(0, 1, 2) -> (2, 0, 1)
       shape:  (2, 4, 3) -> (3, 2, 4)
       sizes:  (12, 3, 1) -> (8, 4, 1)

       value:
                 1, 1, 6, 8,
                 10, 13, 16, 19,

                 2, 4, 0, 9,
                 11, 14, 17, 20,

                 3, 5, 7, 2,
                 12, 15, 18, 21

    */

    Tensor y = x.permute(2, 0, 1);
    y.printShape();
    y.printValue();

    /* validate */
    Tensor y_(3, 2, 4);
    for (int i = 0; i < x.shape[0]; i++) {
        for (int j = 0; j < x.shape[1]; j++) {
            for (int k = 0; k < x.shape[2]; k++) {
                y_(k, i, j) = x(i, j, k);
            }
        }
    }
    y_.printShape();
    y_.printValue();

    /* matrix transpose */
    Tensor x1({2, 3}, {1, 2, 3,
                       4, 5, 6});
    Tensor y1 = x1.permute(1, 0);
    y1.printShape();
    y1.printValue();
    return;
}

struct Person {
    int age;
    std::string name;
    Person():age(0),name("fool"){}
    Person(int age_, const std::string &name_)
        :age(age_),name(name_){}
    void print()
    {
        std::cout<<"person:"<<"name:"<<name<<",age:"<<age<<std::endl;
        return;
    }
};
struct Stu {
    std::string name;
    int rank;
    int score;
    Stu():name("fool"),rank(0),score(0){}
    Stu(const std::string &name_, int rank_, int score_):
        name(name_),rank(rank_),score(score_){}
    void print()
    {
        std::cout<<"Student:"<<"name:"<<name<<",rank:"<<rank<<",score:"<<score<<std::endl;
        return;
    }
};

struct Employee {
    std::string name;
    int salary;
    Employee():name("fool"),salary(0){}
    Employee(const std::string &name_, int salary_)
        :name(name_),salary(salary_){}
    void print()
    {
        std::cout<<"Employee:"<<"name:"<<name<<",salary:"<<salary<<std::endl;
        return;
    }
};

template<typename ...T>
class NameList
{
public:
    using Expand = int[];
    using List = std::tuple<T...>;
    constexpr static int Ni = sizeof... (T);
    List list;
public:
    template<typename Tuple, int N>
    struct Print {
        static void impl(Tuple& value)
        {
            Print<Tuple, N - 1>::impl(value);
            std::get<N - 1>(value).print();
            return;
        }
    };
    template<typename Tuple>
    struct Print<Tuple, 1> {
        static void impl(Tuple& value)
        {
            std::get<0>(value).print();
            return;
        }
    };

    template<typename Tuple, int N>
    struct InversePrint {
        static void impl(Tuple& value)
        {
            std::get<N - 1>(value).print();
            InversePrint<Tuple, N - 1>::impl(value);
            return;
        }
    };
    template<typename Tuple>
    struct InversePrint<Tuple, 1> {
        static void impl(Tuple& value)
        {
            std::get<0>(value).print();
            return;
        }
    };

    template<typename Tuple, int N>
    struct Foreach {
        static void impl(Tuple& value)
        {
            Foreach<Tuple, N - 1>::impl(value);
            std::cout<<N - 1<<","<<N - 2<<std::endl;
            std::cout<<"--------"<<std::endl;
            return;
        }
    };
    template<typename Tuple>
    struct Foreach<Tuple, 1> {
        static void impl(Tuple& value)
        {
            std::cout<<"0"<<std::endl;
            return;
        }
    };
public:
    NameList(){}
    NameList(T&& ...type)
        :list(std::make_tuple<T...>(std::forward<T>(type)...)){}

    void print()
    {
        Print<List, Ni>::impl(list);
        return;
    }
    void testIndex()
    {
        Foreach<List, Ni>::impl(list);
        return;
    }
    void inversePrint()
    {
        InversePrint<List, Ni>::impl(list);
        return;
    }
    template<std::size_t ...i>
    void display_(std::index_sequence<i...>)
    {
        Expand{(std::get<i>(list).print(),0)...};
        return;
    }
    void display()
    {
        display_(std::make_index_sequence<Ni>());
        return;
    }
};

void test_tuple()
{
    NameList<Person, Stu, Employee, Person, Stu, Employee>
            list(Person(12, "person"),
                 Stu("jim", 5, 90),
                 Employee("tom", 1000),
                 Person(120, "fool"),
                 Stu("jimmy", 50, 0),
                 Employee("tommy", 10000));
    //list.print();
    //list.inversePrint();
    list.display();
    NameList<Person, Stu, Employee> men;
    auto& p = std::get<0>(men.list);
    p.age = 1000;
    p.name = "richard123";
    //men.print();
    return;
}


int main()
{
#if 0
    test_lu();
    test_qr();
    test_det();
    test_svd();
    test_kmeans();
    test_conv();
#endif
    test_permute();
    return 0;
}
