#include <iostream>
#include "csv.h"
#include "kmeans.h"
#include "svm.h"
#include "gmm.h"
#include "mat.h"

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
    Mat::LU::inv(x, xi);
    xi.show();
    std::cout<<"test inverse:"<<std::endl;
    Mat I(x.rows, x.cols);
    Mat::mul(I, x, xi);
    I.show();
    return;
}

void test_det()
{
    float value;
    Mat x1(3, 3, {1, 1, 1,
                  1, 2, 3,
                  1, 5, 1});
    Mat::det(x1, value);
    std::cout<<"det:"<<value<<std::endl;



    Mat x2(4, 4, {1, 1, 1, 2,
                  1, 2, 3, 0,
                  0, 5, 1, -1,
                  1, 0, -3, 1});
    Mat::det(x2, value);
    std::cout<<"det:"<<value<<std::endl;
    return;
}

int main()
{
    test_det();
    return 0;
}
