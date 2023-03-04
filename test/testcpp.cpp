#include <tuple>
#include <iostream>
#include <string>
#include <vector>
#include "../utils/csv.h"
#include "metautil.h"

struct Person {
    int age;
    std::string name;
    Person():age(0),name("fool"){}
    Person(int age_, const std::string &name_)
        :age(age_),name(name_){}
    void print() const
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
    void print() const
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
    void print() const
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
    std::cout<<"index travel:"<<std::endl;
    foreachTuple(men.list, [](auto && item){
        item.print();
    });
    return;
}


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
    //test_csv();
    test_tuple();
    return 0;
}
