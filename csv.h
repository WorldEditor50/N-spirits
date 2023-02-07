#ifndef CSV_H
#define CSV_H
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <functional>

template<typename DataType>
class CSV
{
public:
    std::string fileName;
    std::size_t cols;
public:
    CSV():cols(0){}
    CSV(const std::string &fileName_):fileName(fileName_){}
    void clear()
    {
        std::ofstream file(fileName, std::ios::out|std::ios::trunc);
        if (file.is_open() == false) {
            return;
        }
        file.flush();
        file.close();
        return;
    }

    int load(std::string &title, std::vector<DataType> &datas)
    {
        std::ifstream file(fileName, std::ios::in);
        if (file.is_open() == false) {
            return -1;
        }
        /* skip first row */
        std::string line;
        std::getline(file, title);
        cols = 1;
        for (std::size_t i = 0; i < title.size(); i++) {
            if (title[i] == ',') {
                cols++;
            }
        }
        /* read data */
        while (std::getline(file, line)) {
            std::istringstream dataString(line);
            DataType data;
            DataType::parse(dataString, cols, data);
            datas.push_back(data);
        }
        file.close();
        return 0;
    }

    int save(const std::string &title, const std::vector<DataType> &datas)
    {
        std::ofstream file(fileName, std::ios::out);
        if (file.is_open() == false) {
            return -1;
        }
        /* write title */
        file<<title<<std::endl;
        /* write data */
        for (std::size_t i = 0; i < datas.size(); i++) {
            std::string line;
            DataType::toString(datas[i], line);
            file<<line<<std::endl;
        }
        file.close();
        return 0;
    }

    int read(int lineNo, DataType &data)
    {
        std::ifstream file(fileName, std::ios::in);
        if (file.is_open() == false) {
            return -1;
        }
        /* skip first row */
        std::string line;
        std::getline(file, line);
        cols = 1;
        for (std::size_t i = 0; i < line.size(); i++) {
            if (line[i] == ',') {
                cols++;
            }
        }
        /* read data */
        int lineNo_ = 1;
        while (std::getline(file, line)) {
            if (lineNo_ != lineNo) {
                lineNo_++;
                continue;
            } else {
                std::istringstream dataString(line);
                DataType::parse(dataString, cols, data);
                break;
            }
        }
        file.close();
        return 0;
    }

    int find(std::vector<DataType> &datas, std::function<bool(const std::string& line)> filter)
    {
        std::ifstream file(fileName, std::ios::in);
        if (file.is_open() == false) {
            return -1;
        }
        /* skip first row */
        std::string line;
        std::getline(file, line);
        cols = 1;
        for (std::size_t i = 0; i < line.size(); i++) {
            if (line[i] == ',') {
                cols++;
            }
        }
        /* read data */
        while (std::getline(file, line)) {
            if (filter(line) == false) {
                continue;
            }
            std::istringstream dataString(line);
            DataType data;
            DataType::parse(dataString, data);
            datas.push_back(data);
        }
        file.close();
        return 0;
    }

    int writeTitle(const std::string &title)
    {
        std::ofstream file(fileName, std::ios::out);
        if (file.is_open() == false) {
            return -1;
        }
        /* write title */
        file<<title<<std::endl;
        file.close();
        return 0;
    }

    int append(const DataType &data)
    {
        std::ofstream file(fileName, std::ios::out);
        if (file.is_open() == false) {
            return -1;
        }
        /* write data */
        std::string line;
        DataType::toString(data, line);
        file<<line<<std::endl;
        file.close();
        return 0;
    }

    int append(const std::vector<DataType> &data)
    {
        std::ofstream file(fileName, std::ios::out);
        if (file.is_open() == false) {
            return -1;
        }
        /* write data */
        for (std::size_t i = 0; i < data.size(); i++) {
            std::string line;
            DataType::toString(data[i], line);
            file<<line<<std::endl;
        }
        file.close();
        return 0;
    }

    void remove(int lineNo)
    {
        std::string title;
        std::vector<DataType> datas;
        /* load data */
        int ret = load(title, datas);
        if (ret < 0) {
            return;
        }
        /* remove line */
        datas.erase(datas.begin() + lineNo);
        /* empty file */
        clear();
        /* rewrite data */
        save(title, datas);
        return;
    }

    void update(int lineNo, DataType &data)
    {
        std::string title;
        std::vector<DataType> datas;
        /* load data */
        int ret = load(title, datas);
        if (ret < 0) {
            return;
        }
        /* update line */
        std::string line;
        DataType::toString(data, line);
        datas[lineNo] = line;
        /* empty file */
        clear();
        /* rewrite data */
        save(title, datas);
        return;
    }
};

class Strings : public std::vector<std::string>
{
public:
    static void parse(std::istringstream &stream, std::size_t cols, Strings &x)
    {
        for (std::size_t i = 0; i < cols; i++) {
            std::string data;
            std::getline(stream, data, ',');
            x.push_back(data);
        }
        return;
    }

    static void toString(const Strings &data, std::string &line)
    {
        for (std::size_t i = 0; i < data.size(); i++) {
            line += data[i];
            if (i < data.size() - 1) {
                line += ",";
            }
        }
        return;
    }
};

#endif // CSV_H
