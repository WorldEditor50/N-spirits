#include <iostream>
#include "csv.h"
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

int main()
{
    KMeans model;
    SVM<Kernel::RBF> svm;
    GMM gmm;
    Vec x({1, 2, 3, 4, 5});
    return 0;
}
